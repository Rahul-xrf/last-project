"""
SUMO Visual Demo - Watch vehicles moving on the urban network.
This opens SUMO-GUI and runs the simulation slowly so you can
see the electric vehicles driving through the 4x4 urban grid.
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
        print("Set it to your SUMO installation, e.g.:")
        print('  set SUMO_HOME=C:\\Program Files (x86)\\Eclipse\\Sumo')
        sys.exit(1)

    sumo_binary = os.path.join(sumo_home, 'bin', 'sumo-gui')
    sumo_cfg = os.path.join('sumo_config', 'urban.sumocfg')

    if not os.path.exists(sumo_cfg):
        print(f"ERROR: SUMO config not found: {sumo_cfg}")
        sys.exit(1)

    print("=" * 60)
    print("  SUMO Visual Demo - Dynamic Spectrum Allocation")
    print("=" * 60)
    print()
    print("  Opening SUMO-GUI with urban traffic network...")
    print("  - 4x4 grid network with traffic lights")
    print("  - 10 electric vehicles (shown in blue)")
    print()
    print("  Controls:")
    print("    - Simulation runs automatically")
    print("    - Scroll wheel = zoom in/out")
    print("    - Click + drag = pan the view")
    print("    - Click a vehicle to see its details")
    print("    - Press Ctrl+C in terminal to stop")
    print()

    # Start SUMO-GUI (--delay adds ms delay between steps for visibility)
    sumo_cmd = [
        sumo_binary,
        '-c', sumo_cfg,
        '--start',              # Auto-start simulation
        '--delay', '200',       # 200ms delay between steps (visible speed)
        '--window-size', '1200,800',
    ]

    print("  Starting SUMO-GUI...")
    traci.start(sumo_cmd)
    print("  SUMO-GUI is running! Watch the vehicles move.")
    print()

    try:
        step = 0
        while step < 500:  # Run for 500 time steps
            traci.simulationStep()
            vehicles = traci.vehicle.getIDList()

            if step % 25 == 0 and vehicles:
                print(f"  Step {step:4d} | Vehicles: {len(vehicles):2d} | ", end="")
                # Show first vehicle info
                v = vehicles[0]
                pos = traci.vehicle.getPosition(v)
                speed = traci.vehicle.getSpeed(v)
                edge = traci.vehicle.getRoadID(v)
                print(f"{v}: pos=({pos[0]:.0f},{pos[1]:.0f}) "
                      f"speed={speed:.1f}m/s edge={edge}")

            step += 1
            time.sleep(0.05)  # Additional delay for smooth viewing

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
