"""
SUMO Visual Demo - Watch vehicles moving on a real road network.
Opens SUMO-GUI with OpenStreetMap data for realistic visualization.
"""

import os
import sys
import time

try:
    import traci
except ImportError:
    print("ERROR: traci not found. Make sure SUMO is installed.")
    sys.exit(1)


def run_demo():
    sumo_home = os.environ.get('SUMO_HOME', '')
    if not sumo_home:
        print("ERROR: SUMO_HOME environment variable not set.")
        sys.exit(1)

    sumo_binary = os.path.join(sumo_home, 'bin', 'sumo-gui')

    # Check for OSM network, fallback to grid
    osm_net = os.path.join('sumo_config', 'osm_network.net.xml')
    osm_routes = os.path.join('sumo_config', 'osm_routes.rou.xml')
    osm_poly = os.path.join('sumo_config', 'osm_buildings.poly.xml')

    if os.path.exists(osm_net) and os.path.exists(osm_routes):
        net_file = osm_net
        route_file = osm_routes
        print("  Network: OpenStreetMap (real roads)")
    else:
        net_file = os.path.join('sumo_config', 'urban.net.xml')
        route_file = os.path.join('sumo_config', 'urban.rou.xml')
        osm_poly = None
        print("  Network: 4x4 Urban Grid")

    print("=" * 60)
    print("  SUMO Visual Demo - Dynamic Spectrum Allocation")
    print("=" * 60)
    print()
    print("  Controls:")
    print("    - Scroll wheel = zoom in/out")
    print("    - Click + drag = pan the view")
    print("    - Click a vehicle to see its details")
    print("    - Press Ctrl+C in terminal to stop")
    print()

    # Build command
    sumo_cmd = [
        sumo_binary,
        '-n', net_file,
        '-r', route_file,
        '--start',
        '--delay', '150',
        '--window-size', '1200,800',
        '--no-warnings',
        '-e', '500',
    ]

    # Try with buildings, fallback without
    use_buildings = False
    if osm_poly and os.path.exists(osm_poly):
        sumo_cmd_with_poly = sumo_cmd + ['-a', osm_poly]
        try:
            print("  Loading buildings from OSM...")
            traci.start(sumo_cmd_with_poly)
            use_buildings = True
        except Exception:
            print("  Buildings failed, loading without...")
            try:
                traci.close()
            except Exception:
                pass
            traci.start(sumo_cmd)
    else:
        traci.start(sumo_cmd)

    print("  SUMO-GUI is running! Watch the vehicles move.")
    if use_buildings:
        print("  Buildings: Loaded")
    print()

    try:
        step = 0
        while step < 500:
            traci.simulationStep()
            vehicles = traci.vehicle.getIDList()

            if step % 25 == 0:
                vcount = len(vehicles)
                if vehicles:
                    v = vehicles[0]
                    pos = traci.vehicle.getPosition(v)
                    speed = traci.vehicle.getSpeed(v)
                    print(f"  Step {step:4d} | Vehicles: {vcount:2d} | "
                          f"{v}: pos=({pos[0]:.0f},{pos[1]:.0f}) "
                          f"speed={speed:.1f}m/s")
                else:
                    print(f"  Step {step:4d} | Vehicles: {vcount:2d}")

            step += 1
            time.sleep(0.03)

    except KeyboardInterrupt:
        print("\n  Stopped by user.")
    except traci.exceptions.FatalTraCIError:
        print("\n  SUMO-GUI was closed.")
    finally:
        try:
            traci.close()
        except Exception:
            pass

    print()
    print("  Demo finished!")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
