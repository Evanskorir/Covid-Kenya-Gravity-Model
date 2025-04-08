from src.coordinatesloader import CoordinateLoader
from src.dataloader import DataLoader
from src.distance_calculator import DistanceCalculator
from src.gravity_calculator import GravityModel


def main():
    data = DataLoader()

    coord_loader = CoordinateLoader()
    distance_calc = DistanceCalculator(coord_loader.get_all_coordinates())
    distance_calc.compute_all_distances()

    gravity = GravityModel(
        cases_dict=data.get_all_cases(),
        population_dict=data.get_all_population(),
        distance_calculator=distance_calc
    )
    gravity.compute_gravity()

    print("Gravity Nairobi → Kisumu:", gravity.get_gravity("Nairobi", "Kisumu"))

    print("\nTop 5 gravity flows:")
    top5 = sorted(gravity.get_all_gravity().items(), key=lambda x: x[1],
                  reverse=True)[:5]
    for (i, j), score in top5:
        print(f"{i} → {j}: {score:,.2f}")


if __name__ == '__main__':
    main()