import re

filepath = r'd:\last project\project\sumo_config\routes.rou.xml'
with open(filepath, 'r') as f:
    content = f.read()

content = re.sub(r'<vehicle id="(\d+)" depart=', r'<vehicle id="\1" type="car" depart=', content)

with open(filepath, 'w') as f:
    f.write(content)

count = content.count('type="car"')
print(f'Updated {count} vehicles to use car type')
