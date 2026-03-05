"""Download OSM data and convert to SUMO network."""
import urllib.request
import os
import subprocess
import socket

def download_osm():
    """Download road network from OpenStreetMap."""
    # Small urban area in New Delhi (Connaught Place area)
    south, west = 28.6300, 77.2170
    north, east = 28.6350, 77.2220
    
    query = (
        '[out:xml][bbox:{},{},{},{}];'
        '(way["highway"~"^(motorway|trunk|primary|secondary|tertiary|residential|unclassified)$"];>;);'
        'out body;'
    ).format(south, west, north, east)
    
    url = 'https://overpass-api.de/api/interpreter'
    data = ('data=' + query).encode('utf-8')
    output = os.path.join('sumo_config', 'osm_map.osm')
    
    print('Downloading road network from OpenStreetMap...')
    print(f'  Area: New Delhi (Connaught Place)')
    print(f'  Bounds: ({south},{west}) to ({north},{east})')
    
    socket.setdefaulttimeout(60)
    req = urllib.request.Request(url, data=data)
    req.add_header('Content-Type', 'application/x-www-form-urlencoded')
    
    response = urllib.request.urlopen(req, timeout=60)
    content = response.read()
    
    with open(output, 'wb') as f:
        f.write(content)
    
    size = os.path.getsize(output)
    print(f'  Downloaded: {output} ({size/1024:.1f} KB)')
    return output

def convert_to_sumo(osm_file):
    """Convert OSM file to SUMO network."""
    output = os.path.join('sumo_config', 'osm_network.net.xml')
    
    sumo_home = os.environ.get('SUMO_HOME', '')
    netconvert = os.path.join(sumo_home, 'bin', 'netconvert')
    
    print('Converting OSM to SUMO network...')
    
    cmd = [
        netconvert,
        '--osm-files', osm_file,
        '--output-file', output,
        '--geometry.remove',
        '--ramps.guess',
        '--junctions.join',
        '--tls.guess-signals',
        '--tls.discard-simple',
        '--tls.join',
        '--tls.default-type', 'actuated',
        '--no-turnarounds',
        '--no-warnings',
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f'  netconvert stderr: {result.stderr[:500]}')
    
    if os.path.exists(output):
        size = os.path.getsize(output)
        print(f'  Network created: {output} ({size/1024:.1f} KB)')
        return output
    else:
        print('  ERROR: Network file not created')
        return None

def generate_routes(net_file):
    """Generate vehicle routes on the OSM network."""
    sumo_home = os.environ.get('SUMO_HOME', '')
    random_trips = os.path.join(sumo_home, 'tools', 'randomTrips.py')
    route_file = os.path.join('sumo_config', 'osm_routes.rou.xml')
    
    print('Generating vehicle routes...')
    
    cmd = [
        'python', random_trips,
        '-n', net_file,
        '-r', route_file,
        '-b', '0',
        '-e', '100',
        '-p', '5',
        '--seed', '42',
        '--validate',
        '--vehicle-class', 'passenger',
        '--vclass', 'passenger',
        '--trip-attributes', 'type="ev"',
        '--additional-file', os.path.join('sumo_config', 'osm_vtypes.add.xml'),
    ]
    
    # Create vehicle type file
    vtype_file = os.path.join('sumo_config', 'osm_vtypes.add.xml')
    with open(vtype_file, 'w') as f:
        f.write('<additional>\n')
        f.write('    <vType id="ev" accel="2.6" decel="4.5" sigma="0.5" ')
        f.write('length="5" minGap="2.5" maxSpeed="15" ')
        f.write('color="0,128,255" guiShape="passenger/sedan"/>\n')
        f.write('</additional>\n')
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if os.path.exists(route_file):
        size = os.path.getsize(route_file)
        print(f'  Routes created: {route_file} ({size/1024:.1f} KB)')
        return route_file
    else:
        print(f'  stderr: {result.stderr[:500]}')
        print('  Falling back to simple route generation...')
        return generate_simple_routes(net_file, route_file)

def generate_simple_routes(net_file, route_file):
    """Fallback route generation."""
    sumo_home = os.environ.get('SUMO_HOME', '')
    random_trips = os.path.join(sumo_home, 'tools', 'randomTrips.py')
    
    cmd = [
        'python', random_trips,
        '-n', net_file,
        '-r', route_file,
        '-b', '0', '-e', '100',
        '-p', '5', '--seed', '42',
        '--validate',
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if os.path.exists(route_file):
        size = os.path.getsize(route_file)
        print(f'  Routes created (simple): {route_file} ({size/1024:.1f} KB)')
        return route_file
    print(f'  ERROR: {result.stderr[:300]}')
    return None

def create_sumo_config(net_file, route_file):
    """Create SUMO configuration file for OSM network."""
    cfg_file = os.path.join('sumo_config', 'osm.sumocfg')
    
    # Use just filenames since they're in the same directory
    net_basename = os.path.basename(net_file)
    route_basename = os.path.basename(route_file)
    
    with open(cfg_file, 'w') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<configuration>\n')
        f.write('    <input>\n')
        f.write(f'        <net-file value="{net_basename}"/>\n')
        f.write(f'        <route-files value="{route_basename}"/>\n')
        f.write('    </input>\n')
        f.write('    <time>\n')
        f.write('        <begin value="0"/>\n')
        f.write('        <end value="10000"/>\n')
        f.write('        <step-length value="1.0"/>\n')
        f.write('    </time>\n')
        f.write('    <processing>\n')
        f.write('        <collision.action value="warn"/>\n')
        f.write('        <time-to-teleport value="-1"/>\n')
        f.write('    </processing>\n')
        f.write('</configuration>\n')
    
    print(f'  Config created: {cfg_file}')
    return cfg_file


if __name__ == '__main__':
    print('=' * 60)
    print('  OpenStreetMap to SUMO Network Converter')
    print('=' * 60)
    print()
    
    osm_file = download_osm()
    net_file = convert_to_sumo(osm_file)
    
    if net_file:
        route_file = generate_routes(net_file)
        if route_file:
            cfg_file = create_sumo_config(net_file, route_file)
            print()
            print('  All done! Run with:')
            print(f'    sumo-gui -c {cfg_file}')
            print()
            print('=' * 60)
