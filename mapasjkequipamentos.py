# -*- coding: utf-8 -*-
# App Streamlit: Login + Mapa de Obras (PE/PB/AL)
# - CRUD (adicionar/editar/excluir)
# - Geocode resiliente (Nominatim) + clique manual + lat/lon manual
# - Foco inicial na RMR (Recife e vizinhos)
# - Status (inclui "Obra concluída")
# - Exportar CSV
# - Secrets do Streamlit (APP_USERS_JSON, COOKIE_KEY, USE_GSHEETS, GSPREAD_SERVICE_ACCOUNT, GSHEETS_URL, GSHEETS_WORKSHEET)
# - Persistência: CSV local (efêmero) OU Google Sheets (persistente)
# - Compatível com Streamlit Cloud

from pathlib import Path
from datetime import datetime
import json
import os
import re

import pandas as pd
import streamlit as st

# autenticação (opcionalmente com streamlit-authenticator)
try:
    import streamlit_authenticator as stauth
    HAS_STAUTH = True
except Exception:
    HAS_STAUTH = False

# gspread (opcional para Google Sheets)
try:
    import gspread
    from google.oauth2.service_account import Credentials
    HAS_GSHEETS = True
except Exception:
    HAS_GSHEETS = False

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

# -----------------------------------------------------------------------------
# Config e utilidades de secrets/ambiente
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Mapa de Obras (PE, PB, AL)", layout="wide")

def get_secret(key: str, default: str = ""):
    """Lê primeiro em st.secrets (Streamlit Cloud), depois em os.environ."""
    try:
        if key in st.secrets:
            val = st.secrets[key]
            # secrets pode ser dict; converte para JSON string quando precisamos de string
            if isinstance(val, (dict, list)):
                return json.dumps(val, ensure_ascii=False)
            return str(val)
    except Exception:
        pass
    return os.getenv(key, default)

# diretório local (na nuvem é efêmero; para persistir use Google Sheets)
DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True, parents=True)
CSV_PATH = DATA_DIR / "obras.csv"

ESTADOS_PERMITIDOS = {"PE", "PB", "AL"}

# mapa/zoom
MAP_CENTER = (-8.50, -36.80)
MAP_ZOOM = 7
RMR_BOUNDS = ((-8.50, -35.35), (-7.70, -34.75))
RECIFE_CENTER = (-8.05428, -34.8813)

# viewbox Nordeste para geocoder (SW, NE)
VIEWBOX_NE = [(-10.8, -42.5), (-6.0, -33.5)]
COUNTRY = "br"

STATUS_OPCOES = {
    "Obra visitada — cliente": "green",
    "Obra visitada — não cliente": "orange",
    "Obra não visitada": "red",
    "Obra concluída": "blue",
}

# -----------------------------------------------------------------------------
# Persistência via Google Sheets (opcional)
# -----------------------------------------------------------------------------
def gsheets_enabled() -> bool:
    return (get_secret("USE_GSHEETS", "false").lower() == "true") and HAS_GSHEETS

def _gs_client():
    svc_json = get_secret("GSPREAD_SERVICE_ACCOUNT", "")
    if not svc_json:
        raise RuntimeError("Faltou GSPREAD_SERVICE_ACCOUNT nos Secrets.")
    try:
        info = json.loads(svc_json)  # string JSON
        if isinstance(info, str):
            info = json.loads(info)
    except Exception as e:
        raise RuntimeError("GSPREAD_SERVICE_ACCOUNT inválido.") from e
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(info, scopes=scopes)
    return gspread.authorize(creds)

def _gs_open():
    url = get_secret("GSHEETS_URL", "")
    if not url:
        raise RuntimeError("Faltou GSHEETS_URL nos Secrets.")
    ws_name = get_secret("GSHEETS_WORKSHEET", "obras")
    gc = _gs_client()
    sh = gc.open_by_url(url)
    try:
        ws = sh.worksheet(ws_name)
    except gspread.exceptions.WorksheetNotFound:
        ws = sh.add_worksheet(title=ws_name, rows=2, cols=20)
        ws.update([[
            "id","nome_obra","contato","telefone","endereco","cidade",
            "estado","status","lat","lon","criado_em"
        ]])
    return ws

def gs_load_df() -> pd.DataFrame:
    ws = _gs_open()
    values = ws.get_all_values()
    if not values:
        return pd.DataFrame(columns=[
            "id","nome_obra","contato","telefone","endereco","cidade",
            "estado","status","lat","lon","criado_em"
        ])
    header, *rows = values
    df = pd.DataFrame(rows, columns=header)
    # tipos
    for c in ("lat","lon"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # garante id como str
    if "id" in df.columns:
        df["id"] = df["id"].astype(str)
    return df

def gs_save_df(df: pd.DataFrame):
    ws = _gs_open()
    if df.empty:
        ws.clear()
        ws.update([[
            "id","nome_obra","contato","telefone","endereco","cidade",
            "estado","status","lat","lon","criado_em"
        ]])
        return
    out = df.copy()
    # garante colunas e ordem
    cols = ["id","nome_obra","contato","telefone","endereco","cidade",
            "estado","status","lat","lon","criado_em"]
    for c in cols:
        if c not in out.columns:
            out[c] = ""
    out = out[cols]
    # tudo string pro update
    out = out.fillna("").astype(str)
    ws.clear()
    ws.update([cols] + out.values.tolist())

# -----------------------------------------------------------------------------
# Autenticação
# -----------------------------------------------------------------------------
def _is_bcrypt_hash(s: str) -> bool:
    return isinstance(s, str) and s.startswith(("$2a$", "$2b$", "$2y$"))

def _load_users_from_secrets() -> dict:
    """APP_USERS_JSON pode ser JSON string ou dict em secrets."""
    raw = get_secret("APP_USERS_JSON", "")
    if not raw:
        return {"demo": "demo"}  # fallback p/ teste
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except Exception:
        pass
    return {"demo": "demo"}

def _hash_pw_compat(pw: str) -> str:
    """Tenta várias APIs do streamlit-authenticator p/ compatibilidade."""
    try:
        return stauth.Hasher([pw]).generate()[0]
    except Exception:
        try:
            return stauth.Hasher.hash(pw)
        except Exception:
            try:
                from streamlit_authenticator.utilities.hasher import Hasher as _Hasher
                return _Hasher.hash(pw)
            except Exception:
                return pw  # fallback plaintext (auto_hash=True depois)

USERS_ENV = _load_users_from_secrets()
COOKIE_KEY = get_secret("COOKIE_KEY", "mude-este-cookie-key-bem-grande")
USE_AUTHENTICATOR = get_secret("USE_AUTHENTICATOR", "true").lower() == "true"

def do_login_with_authenticator():
    """Compatível com versões novas/antigas do streamlit-authenticator."""
    CREDENTIALS = {"usernames": {}}
    any_plaintext = False
    for u, pw in USERS_ENV.items():
        display = u.split("@")[0]
        if _is_bcrypt_hash(pw):
            hashed = pw
        else:
            try:
                hashed = _hash_pw_compat(pw)
                if hashed == pw:
                    any_plaintext = True
            except Exception:
                hashed = pw
                any_plaintext = True
        CREDENTIALS["usernames"][u] = {"name": display, "password": hashed}

    auto_hash_flag = any_plaintext
    try:
        authenticator = stauth.Authenticate(
            CREDENTIALS, cookie_name="obras_auth",
            key=COOKIE_KEY, cookie_expiry_days=30, auto_hash=auto_hash_flag
        )
    except TypeError:
        authenticator = stauth.Authenticate(
            CREDENTIALS, cookie_name="obras_auth",
            cookie_key=COOKIE_KEY, cookie_expiry_days=30, auto_hash=auto_hash_flag
        )

    result = None
    try:
        result = authenticator.login(location="main", fields={"Form name": "Login"})
    except TypeError:
        result = authenticator.login("Login", "main")

    name = st.session_state.get("name")
    auth_status = st.session_state.get("authentication_status")
    username = st.session_state.get("username")

    # Se versão antiga retornou tupla
    if isinstance(result, tuple) and len(result) == 3:
        name, auth_status, username = result

    if auth_status:
        with st.sidebar:
            st.caption(f"Logado como **{name}**")
            try:
                authenticator.logout("Sair", "sidebar")
            except TypeError:
                authenticator.logout("Sair", "sidebar")

    return name, auth_status, username

# Gate de autenticação:
if USE_AUTHENTICATOR and HAS_STAUTH:
    name, auth_status, username = do_login_with_authenticator()
    if auth_status is False:
        st.error("Usuário/senha inválidos."); st.stop()
    elif auth_status is None:
        st.warning("Informe suas credenciais."); st.stop()
else:
    # Modo simples (útil localmente). Em produção, mantenha o app privado na Streamlit Cloud.
    st.info("Login simples ativo (instale 'streamlit-authenticator' para segurança melhor ou USE_AUTHENTICATOR=false).")
    if "auth_ok" not in st.session_state:
        st.session_state.auth_ok = False
        st.session_state.username = ""
    if not st.session_state.auth_ok:
        with st.form("login_simplificado"):
            u = st.text_input("Usuário")
            p = st.text_input("Senha", type="password")
            ok = st.form_submit_button("Entrar")
        if ok and u in USERS_ENV and USERS_ENV[u] == p:
            st.session_state.auth_ok = True
            st.session_state.username = u
            st.rerun()
        elif ok:
            st.error("Usuário/senha inválidos."); st.stop()
    else:
        with st.sidebar:
            st.caption(f"Logado como **{st.session_state.username}**")
            if st.button("Sair"):
                st.session_state.clear()
                st.rerun()

# -----------------------------------------------------------------------------
# Geocode / utilidades
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def make_geolocator():
    geo = Nominatim(user_agent="jk-obras-app")
    return {
        "geo": geo,
        "geocode": RateLimiter(geo.geocode, min_delay_seconds=1),
        "reverse": RateLimiter(geo.reverse, min_delay_seconds=1),
    }

def _norm(s: str) -> str:
    return (s or "").strip()

def _uf_from_address_dict(addr: dict) -> str | None:
    if not addr:
        return None
    uf = (addr.get("state_code") or addr.get("state") or "").strip().upper()
    mapa = {"PERNAMBUCO": "PE", "PARAÍBA": "PB", "PARAIBA": "PB", "ALAGOAS": "AL"}
    return mapa.get(uf, uf) if uf else None

def geocode_resiliente(endereco: str, cidade: str | None = None, uf_hint: str | None = None):
    g = make_geolocator()
    geocode = g["geocode"]
    cand = []
    endereco = _norm(endereco)
    cidade = _norm(cidade)
    if uf_hint:
        uf_hint = _norm(uf_hint).upper()

    if endereco:
        cand.append(endereco)
        if uf_hint and uf_hint in ESTADOS_PERMITIDOS and uf_hint not in endereco.upper():
            cand.append(f"{endereco}, {uf_hint}, Brasil")
        if cidade and cidade.upper() not in endereco.upper():
            cand.append(f"{endereco}, {cidade}, Brasil")
    if cidade and uf_hint:
        cand.append(f"{cidade}, {uf_hint}, Brasil")
    if uf_hint:
        cand.append(f"{uf_hint}, Brasil")
    if not cidade:
        m = re.search(r",\s*([A-Za-zÀ-ú\s]+),\s*(PE|PB|AL)\b", endereco, flags=re.I)
        if m:
            cand.append(f"{m.group(1).strip()}, {m.group(2).upper()}, Brasil")

    seen = set()
    queries = []
    for q in cand:
        if q and q not in seen:
            queries.append(q)
            seen.add(q)

    for q in queries:
        try:
            loc = geocode(
                q,
                addressdetails=True,
                country_codes=COUNTRY,
                viewbox=VIEWBOX_NE,
                bounded=True,
                timeout=10,
            )
        except Exception:
            loc = None
        if loc:
            uf = _uf_from_address_dict(loc.raw.get("address", {}))
            return {"lat": loc.latitude, "lon": loc.longitude, "uf": uf}
    return None

def reverse_uf(lat: float, lon: float) -> str | None:
    g = make_geolocator()
    rev = g["reverse"]
    try:
        loc = rev((lat, lon), addressdetails=True, language="pt", timeout=10)
        return _uf_from_address_dict(loc.raw.get("address", {})) if loc else None
    except Exception:
        return None

def in_bounds(lat: float, lon: float, bounds) -> bool:
    (south, west), (north, east) = bounds
    return (
        pd.notna(lat) and pd.notna(lon) and south <= lat <= north and west <= lon <= east
    )

# -----------------------------------------------------------------------------
# Carregar/salvar dados (CSV local ou Google Sheets)
# -----------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    if gsheets_enabled():
        try:
            return gs_load_df()
        except Exception as e:
            st.error(f"Erro ao carregar Google Sheets: {e}")
    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH, dtype=str)
        for c in ("lat", "lon"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df
    return pd.DataFrame(
        columns=[
            "id",
            "nome_obra",
            "contato",
            "telefone",
            "endereco",
            "cidade",
            "estado",
            "status",
            "lat",
            "lon",
            "criado_em",
        ]
    )

def save_data(df: pd.DataFrame):
    if gsheets_enabled():
        try:
            gs_save_df(df)
            return
        except Exception as e:
            st.error(f"Erro ao salvar no Google Sheets: {e}")
    df.to_csv(CSV_PATH, index=False, encoding="utf-8")

def next_id(df: pd.DataFrame) -> int:
    if df.empty:
        return 1
    try:
        return int(df["id"].astype(int).max()) + 1
    except Exception:
        return len(df) + 1

def make_popup_html(row: pd.Series) -> str:
    return f"""
    <div style="font-family:Arial; font-size:13px; line-height:1.3;">
      <div style="font-weight:bold; font-size:14px; margin-bottom:6px;">{row.get('nome_obra','')}</div>
      <div><b>Status:</b> {row.get('status','')}</div>
      <div><b>Contato:</b> {row.get('contato','')}</div>
      <div><b>Telefone:</b> {row.get('telefone','')}</div>
      <div><b>Endereço:</b> {row.get('endereco','')}</div>
      <div><b>Cidade/UF:</b> {row.get('cidade','')} / {row.get('estado','')}</div>
      <div style="color:#666;"><i>Inserido em {row.get('criado_em','')}</i></div>
    </div>"""

# -----------------------------------------------------------------------------
# Estado de sessão
# -----------------------------------------------------------------------------
if "df" not in st.session_state:
    st.session_state.df = load_data()
if "pending" not in st.session_state:
    st.session_state.pending = None
if "await_click" not in st.session_state:
    st.session_state.await_click = False
if "focus_rmr" not in st.session_state:
    st.session_state.focus_rmr = True

# -----------------------------------------------------------------------------
# Sidebar — formulário novo registro
# -----------------------------------------------------------------------------
st.sidebar.header("Nova obra")
status_labels = list(STATUS_OPCOES.keys())
default_index = status_labels.index("Obra não visitada")

with st.sidebar.form("form_obra", clear_on_submit=False):
    nome_obra = st.text_input("Nome da obra *", "")
    contato = st.text_input("Contato", "")
    telefone = st.text_input("Telefone", "")
    endereco = st.text_area("Endereço (rua, nº, bairro, cidade, UF) *", "")
    cidade = st.text_input("Cidade (ajuda no geocoder)", "")
    status_label = st.selectbox("Status *", status_labels, index=default_index)
    st.caption("Se o geocoder falhar: clique no mapa ou informe lat/lon.")
    c1, c2 = st.columns(2)
    lat_manual = c1.text_input("Latitude", "")
    lon_manual = c2.text_input("Longitude", "")
    colb1, colb2 = st.columns(2)
    add_btn = colb1.form_submit_button("Adicionar", type="primary")
    limpar_btn = colb2.form_submit_button("Limpar")

if limpar_btn:
    st.rerun()

if add_btn:
    df = st.session_state.df
    if not nome_obra.strip() or not endereco.strip():
        st.sidebar.error("Preencha pelo menos Nome da obra e Endereço.")
    else:
        try:
            lat_m = float(lat_manual) if lat_manual.strip() else None
            lon_m = float(lon_manual) if lon_manual.strip() else None
        except ValueError:
            lat_m = lon_m = None

        uf_in_text = None
        m = re.search(r"\b(PE|PB|AL)\b", endereco.upper())
        if m:
            uf_in_text = m.group(1)

        if lat_m is not None and lon_m is not None:
            uf = reverse_uf(lat_m, lon_m) or (uf_in_text or "")
            novo = {
                "id": next_id(df),
                "nome_obra": nome_obra.strip(),
                "contato": contato.strip(),
                "telefone": telefone.strip(),
                "endereco": endereco.strip(),
                "cidade": _norm(cidade),
                "estado": uf,
                "status": status_label,
                "lat": lat_m,
                "lon": lon_m,
                "criado_em": datetime.now().strftime("%Y-%m-%d %H:%M"),
            }
            df = pd.concat([df, pd.DataFrame([novo])], ignore_index=True)
            st.session_state.df = df
            save_data(df)
            st.sidebar.success("Obra adicionada (lat/lon manual).")
        else:
            geo = geocode_resiliente(endereco, cidade=cidade, uf_hint=uf_in_text)
            if geo:
                novo = {
                    "id": next_id(df),
                    "nome_obra": nome_obra.strip(),
                    "contato": contato.strip(),
                    "telefone": telefone.strip(),
                    "endereco": endereco.strip(),
                    "cidade": _norm(cidade),
                    "estado": geo["uf"] or (uf_in_text or ""),
                    "status": status_label,
                    "lat": geo["lat"],
                    "lon": geo["lon"],
                    "criado_em": datetime.now().strftime("%Y-%m-%d %H:%M"),
                }
                df = pd.concat([df, pd.DataFrame([novo])], ignore_index=True)
                st.session_state.df = df
                save_data(df)
                st.sidebar.success("Obra adicionada ao mapa!")
            else:
                st.session_state.pending = {
                    "id": next_id(df),
                    "nome_obra": nome_obra.strip(),
                    "contato": contato.strip(),
                    "telefone": telefone.strip(),
                    "endereco": endereco.strip(),
                    "cidade": _norm(cidade),
                    "estado": uf_in_text or "",
                    "status": status_label,
                    "lat": None,
                    "lon": None,
                    "criado_em": datetime.now().strftime("%Y-%m-%d %H:%M"),
                }
                st.session_state.await_click = True
                st.warning("Geocoder falhou. Clique no mapa para posicionar o alfinete OU informe lat/lon e envie novamente.")

# -----------------------------------------------------------------------------
# Main — mapa + controles
# -----------------------------------------------------------------------------
st.title("Mapa de Obras — PE • PB • AL")
cols = st.columns([1, 1, 2, 1])
with cols[0]:
    focus_rmr = st.checkbox("Focar na RMR (inicial)", value=st.session_state.focus_rmr)
    st.session_state.focus_rmr = focus_rmr
with cols[1]:
    filtro_uf = st.multiselect("Filtrar UF", sorted(ESTADOS_PERMITIDOS))
with cols[2]:
    filtro_status = st.multiselect("Filtrar status", list(STATUS_OPCOES.keys()))
with cols[3]:
    st.download_button(
        "Baixar dados (.csv)",
        data=st.session_state.df.to_csv(index=False).encode("utf-8"),
        file_name="obras.csv",
        mime="text/csv",
        width="stretch",  # <— atualizado
    )

df_view = st.session_state.df.copy()
if not df_view.empty:
    # segurança: garante colunas
    for c in ["status", "estado", "lat", "lon"]:
        if c not in df_view.columns:
            df_view[c] = None

if filtro_status:
    df_view = df_view[df_view["status"].isin(filtro_status)]
if filtro_uf:
    df_view = df_view[df_view["estado"].isin(filtro_uf)]

def filter_rmr(df_in: pd.DataFrame) -> pd.DataFrame:
    if df_in.empty:
        return df_in
    return df_in[
        df_in.apply(lambda r: in_bounds(r.get("lat"), r.get("lon"), RMR_BOUNDS), axis=1)
    ]

use_rmr = focus_rmr
df_map = filter_rmr(df_view) if use_rmr else df_view

if use_rmr:
    m = folium.Map(location=RECIFE_CENTER, zoom_start=12, tiles="OpenStreetMap")
    if not df_map.empty and df_map["lat"].notna().any():
        lats = df_map["lat"].dropna().tolist()
        lons = df_map["lon"].dropna().tolist()
        sw = [min(lats + [RMR_BOUNDS[0][0]]), min(lons + [RMR_BOUNDS[0][1]])]
        ne = [max(lats + [RMR_BOUNDS[1][0]]), max(lons + [RMR_BOUNDS[1][1]])]
        m.fit_bounds([sw, ne])
    else:
        sw, ne = RMR_BOUNDS
        m.fit_bounds([sw, ne])
else:
    m = folium.Map(location=MAP_CENTER, zoom_start=MAP_ZOOM, tiles="OpenStreetMap")
    cluster = MarkerCluster().add_to(m)

def add_marker(row):
    color = STATUS_OPCOES.get(row["status"], "blue")
    folium.Marker(
        location=(row["lat"], row["lon"]),
        icon=folium.Icon(color=color, icon="info-sign"),
        popup=folium.Popup(make_popup_html(row), max_width=350),
        tooltip=row.get("nome_obra", "Obra"),
    ).add_to(m if use_rmr else cluster)

if not df_map.empty:
    for _, row in df_map.dropna(subset=["lat", "lon"]).iterrows():
        add_marker(row)

if st.session_state.await_click:
    folium.Marker(
        location=RECIFE_CENTER if use_rmr else MAP_CENTER,
        icon=folium.Icon(color="gray", icon="plus"),
        tooltip="Clique no mapa para definir a posição da obra pendente",
    ).add_to(m)

st_data = st_folium(m, height=650, width=None)

if st.session_state.await_click and st.session_state.pending and st_data:
    last = st_data.get("last_clicked")
    if last and "lat" in last and "lng" in last:
        latc, lonc = float(last["lat"]), float(last["lng"])
        uf = reverse_uf(latc, lonc) or st.session_state.pending.get("estado", "")
        novo = st.session_state.pending.copy()
        novo["lat"] = latc
        novo["lon"] = lonc
        novo["estado"] = uf
        st.session_state.df = pd.concat([st.session_state.df, pd.DataFrame([novo])], ignore_index=True)
        save_data(st.session_state.df)
        st.session_state.pending = None
        st.session_state.await_click = False
        st.success("Posição definida pelo clique no mapa. Obra salva!")
        st.rerun()

# -----------------------------------------------------------------------------
# Editar / Excluir
# -----------------------------------------------------------------------------
with st.sidebar.expander("Gerenciar alfinetes (editar/excluir)", expanded=False):
    dfe = st.session_state.df
    if dfe.empty:
        st.info("Nenhum registro ainda.")
    else:
        opcoes = dfe.assign(
            rotulo=lambda d: d.apply(
                lambda r: f"{r['id']} — {r.get('nome_obra','(sem nome)')} ({r.get('estado','')})",
                axis=1,
            )
        )[["id", "rotulo"]]
        escolha = st.selectbox("Selecione uma obra", opcoes["rotulo"].tolist())
        sel_id = int(escolha.split(" — ")[0])
        row = dfe[dfe["id"].astype(int) == sel_id].iloc[0]

        with st.form("form_editar", clear_on_submit=False):
            edit_nome = st.text_input("Nome da obra *", value=row.get("nome_obra", ""))
            edit_contato = st.text_input("Contato", value=row.get("contato", ""))
            edit_tel = st.text_input("Telefone", value=row.get("telefone", ""))
            edit_end = st.text_area(
                "Endereço (rua, nº, bairro, cidade, UF) *", value=row.get("endereco", "")
            )
            edit_city = st.text_input("Cidade", value=row.get("cidade", ""))
            status_idx = (
                status_labels.index(row.get("status", "Obra não visitada"))
                if row.get("status", "Obra não visitada") in status_labels
                else 0
            )
            edit_status = st.selectbox("Status *", status_labels, index=status_idx)

            colE1, colE2 = st.columns(2)
            geocode_on_save = colE1.checkbox("Regeocodificar ao salvar", value=False)
            confirm_id = colE2.text_input(
                "Confirmar ID p/ excluir",
                value="",
                help=f"Digite {sel_id} para excluir.",
            )

            colB1, colB2 = st.columns(2)
            salvar_btn = colB1.form_submit_button("Salvar alterações", type="primary")
            excluir_btn = colB2.form_submit_button("Excluir", type="secondary")

        if salvar_btn:
            if not edit_nome.strip() or not edit_end.strip():
                st.error("Preencha pelo menos Nome da obra e Endereço.")
            else:
                idx = dfe.index[dfe["id"].astype(int) == sel_id][0]
                dfe.at[idx, "nome_obra"] = edit_nome.strip()
                dfe.at[idx, "contato"] = edit_contato.strip()
                dfe.at[idx, "telefone"] = edit_tel.strip()
                dfe.at[idx, "endereco"] = edit_end.strip()
                dfe.at[idx, "cidade"] = (edit_city or "").strip()
                dfe.at[idx, "status"] = edit_status

                if geocode_on_save:
                    uf_hint = None
                    m2 = re.search(r"\b(PE|PB|AL)\b", edit_end.upper())
                    if m2:
                        uf_hint = m2.group(1)
                    geo = geocode_resiliente(edit_end, cidade=(edit_city or "").strip(), uf_hint=uf_hint)
                    if geo:
                        dfe.at[idx, "lat"] = geo["lat"]
                        dfe.at[idx, "lon"] = geo["lon"]
                        dfe.at[idx, "estado"] = geo["uf"] or dfe.at[idx, "estado"]
                    else:
                        st.warning("Não consegui geocodificar; mantive as coordenadas antigas.")

                st.session_state.df = dfe
                save_data(dfe)
                st.success("Alterações salvas.")
                st.rerun()

        if excluir_btn:
            if confirm_id.strip() == str(sel_id):
                dfe = dfe[dfe["id"].astype(int) != sel_id].copy()
                st.session_state.df = dfe
                save_data(dfe)
                st.success(f"Obra ID {sel_id} excluída.")
                st.rerun()
            else:
                st.error(f"Para excluir, digite exatamente o ID {sel_id}.")

# -----------------------------------------------------------------------------
# Tabela
# -----------------------------------------------------------------------------
with st.expander("Ver dados (visualização)"):
    st.dataframe(
        st.session_state.df.sort_values("criado_em", ascending=False),
        width="stretch",  # <— atualizado
    )
