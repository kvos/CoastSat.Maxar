# initialise ARLULA API session
import arlulaapi
api_key = '1r0TQCxiH0fieLaOx0qT4FgmC1wvtcvkCPUqRSgfM8ITd1Juit6LKJVzBlHn'
secret = 'BlPfm7R0ZY2cb0j2zS47SPRVNWXcq7zp8z3vNumuIucCQ60Ir2DkiRDwB042'
arlula_session = arlulaapi.ArlulaSession(api_key, secret)

# check if the order is in the list of all orders
orders = arlula_session.list_orders()
order_id = '4b3857dc-6416-40ac-ac0d-7a7528f6722d'
for order in orders:
    if order.id == order_id:
        print(order.id)
# download the order 
arlula_session.get_order_resources(
    id=order_id,
    folder='demo'
)