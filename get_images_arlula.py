# initialise ARLULA API session
import arlulaapi
api_key = '1r0TQCxiH0fieLaOx0qT4FgmC1wvtcvkCPUqRSgfM8ITd1Juit6LKJVzBlHn'
secret = 'BlPfm7R0ZY2cb0j2zS47SPRVNWXcq7zp8z3vNumuIucCQ60Ir2DkiRDwB042'
arlula_session = arlulaapi.ArlulaSession(api_key, secret)

# order ids from email notifications that Kilian received
order1 = '3aef90ff-9232-4322-b0a3-118b2f12b4cf'
order2 = '1caa0079-1805-4197-bcb2-9ba70ee137a2'
order3 = 'd8b4462c-bbae-47bd-b1f5-e1078e645c60'
order4 = '5fd667da-44dc-4014-95a2-524085a93108'
custom_orders = [order1,order2,order3,order4]

# find these orders from the list of all orders
all_orders = arlula_session.list_orders()
order_list = []
for order in all_orders:
    if order.id in custom_orders:
        order_list.append(order)
        print(order.id)
        
# download the orders
for i,order in enumerate(order_list):
    arlula_session.get_order_resources(
        id=order.id,
        folder='order%d'%(i+1)
    )