"""Storefront pages: catalogue, cart, orders."""
from __future__ import annotations

import streamlit as st

import store
from ui.common import go, money


def shop_page() -> None:
    st.header("Shop")
    products = store.list_products()
    for row in range(0, len(products), 4):
        for col, p in zip(st.columns(4), products[row:row + 4]):
            with col.container(border=True):
                st.markdown(f"## {p['emoji']}")
                st.markdown(f"**{p['name']}**")
                st.caption(p["blurb"])
                st.markdown(money(p["price"]))
                if st.button("Add to cart", key=f"add_{p['id']}"):
                    st.session_state.cart[p["id"]] = st.session_state.cart.get(p["id"], 0) + 1
                    st.toast(f"{p['name']} added")
                    st.rerun()


def cart_page(customer: str) -> None:
    st.header("Cart")
    cart: dict[str, int] = st.session_state.cart
    if not cart:
        st.info("Your cart is empty.")
        st.button("Browse products", on_click=go, args=("Shop",))
        return
    prods = {p["id"]: p for p in store.list_products()}
    total = 0
    for pid, qty in list(cart.items()):
        p = prods[pid]
        c1, c2, c3 = st.columns([4, 2, 1])
        c1.markdown(f"{p['emoji']} **{p['name']}**  \n{money(p['price'])} each")
        new_qty = c2.number_input("Qty", 0, 20, qty, key=f"qty_{pid}")
        if new_qty != qty:
            cart[pid] = new_qty
            if new_qty == 0:
                del cart[pid]
            st.rerun()
        total += p["price"] * qty
        c3.markdown(money(p["price"] * qty))
    st.subheader(f"Total {money(total)}")
    if st.button("Place order", type="primary"):
        order_id = store.create_order(customer, cart)
        st.session_state.cart = {}
        st.success(f"Order {order_id} placed. Expected in about 5 days.")


def orders_page(customer: str) -> None:
    st.header("My orders")
    orders = store.list_orders(customer)
    if not orders:
        st.info("No orders yet.")
    for o in orders:
        with st.expander(f"{o['order_id']} · {o['status']} · {money(o['total'])}"):
            st.write(", ".join(f"{i['qty']}× {i['name']}" for i in o["items"]))
            st.caption(f"Placed {o['created_at']} · expected {o['expected_delivery']}")
            st.button("Report a problem", key=f"help_{o['order_id']}", on_click=go,
                      args=("Support desk", o["order_id"]))
