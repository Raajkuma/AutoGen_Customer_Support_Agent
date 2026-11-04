"""ShopEagle: e-commerce store + AI support desk (live workflow trace, human approval).  Run: streamlit run app.py"""
from __future__ import annotations

import time

import streamlit as st

import auth
import config
import llm
import store
import tickets
from logging_config import setup_logging
from ui import shop, staff, support

st.set_page_config(page_title="ShopEagle", page_icon="🛍️", layout="wide")
setup_logging()
store.init_db()
tickets.init()

CUSTOMER_PAGES = ["Shop", "Cart", "My orders", "Support desk", "Ticket log"]
STAFF_PAGES = ["Approvals", "Workflow trace"]
for key, default in {"cart": {}, "page": "Shop", "support_order": "No order", "role": "customer",
                     "staff_name": "", "last_submit": 0.0, "login_failures": 0, "login_locked_until": 0.0}.items():
    st.session_state.setdefault(key, default)


@st.cache_data(ttl=10)
def model_status() -> tuple[bool, str]:
    return llm.health()


def staff_login_box() -> None:
    with st.expander("Staff login"):
        if not auth.staff_login_enabled():
            st.caption("Set STAFF_PIN in .env to enable staff access.")
            return
        name = st.text_input("Your name", key="login_name")
        pin = st.text_input("PIN", type="password", key="login_pin")
        if st.button("Sign in as staff"):
            if time.time() < st.session_state.login_locked_until:
                st.error("Too many attempts. Try again in a minute.")
            elif name.strip() and auth.verify_staff(pin):
                st.session_state.update(role="staff", staff_name=name.strip(), login_failures=0)
                st.rerun()
            else:
                st.session_state.login_failures += 1
                if st.session_state.login_failures >= 5:
                    st.session_state.login_locked_until = time.time() + 60
                    st.session_state.login_failures = 0
                st.error("Wrong name or PIN.")
        if auth.pin_is_weak():
            st.warning("STAFF_PIN is weak or a placeholder. Change it before sharing this app.")


with st.sidebar:
    st.title("🛍️ ShopEagle")
    customer = st.selectbox("Signed in as", store.CUSTOMERS, help="Demo accounts")
    is_staff = st.session_state.role == "staff"
    pages = CUSTOMER_PAGES + (STAFF_PAGES if is_staff else [])
    if st.session_state.page not in pages:
        st.session_state.page = "Shop"
    st.radio("Go to", pages, key="page")
    st.caption(f"Cart: {sum(st.session_state.cart.values())} item(s)")
    ok, detail = model_status()
    st.markdown(f"{'🟢' if ok else '🔴'} Local model: `{config.MODEL}`")
    if not ok:
        st.caption(detail)
    if is_staff:
        st.success(f"Staff: {st.session_state.staff_name}")
        pending = int((tickets.read_all()["approval"] == "Pending").sum())
        if pending:
            st.warning(f"⏸️ {pending} ticket(s) awaiting approval")
        if st.button("Sign out"):
            st.session_state.update(role="customer", staff_name="")
            st.rerun()
    else:
        staff_login_box()

page = st.session_state.page
if page == "Shop":
    shop.shop_page()
elif page == "Cart":
    shop.cart_page(customer)
elif page == "My orders":
    shop.orders_page(customer)
elif page == "Support desk":
    support.support_page(customer)
elif page == "Ticket log":
    staff.log_page(customer, is_staff)
elif page == "Approvals" and is_staff:
    staff.approvals_page(st.session_state.staff_name)
elif page == "Workflow trace" and is_staff:
    staff.trace_page()
