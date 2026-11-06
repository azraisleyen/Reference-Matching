import json
from uav_refmatch.doctor import doctor

if __name__ == "__main__":
    print(json.dumps(doctor(), indent=2))
