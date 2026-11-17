import numpy as np
import pandas as pd

def diagnose_network(
    adjacency,
    node_order,
    vaccinated,
    beta_values
):
    n = len(node_order)

    node_to_index = {
        node: i
        for i, node in enumerate(node_order)
    }

    vaccinated_indices = {
        node_to_index[node]
        for node in vaccinated
        if node in node_to_index
    }

    degrees = []
    total_durations = []

    for i in range(n):

        contacts = adjacency[i]

        degrees.append(len(contacts))

        total_durations.append(
            sum(duration for _, duration in contacts)
        )

    network_df = pd.DataFrame({
        "participant": node_order,
        "degree": degrees,
        "total_contact_seconds": total_durations,
        "vaccinated": [
            i in vaccinated_indices
            for i in range(n)
        ]
    })

    print("\n===== NETWORK SUMMARY =====")

    print(network_df[
        ["degree", "total_contact_seconds"]
    ].describe(
        percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    ))

    print("\n===== NETWORK STRUCTURE =====")

    print("Participants:", n)
    print("Vaccinated:", len(vaccinated_indices))
    print("Unvaccinated:", n - len(vaccinated_indices))
    print("Total adjacency entries:", sum(degrees))
    print("Mean degree:", np.mean(degrees))
    print("Median degree:", np.median(degrees))
    print("Mean contact duration (seconds):",
          np.mean(total_durations))

    print("\n===== EXPECTED FIRST-GENERATION TRANSMISSION =====")

    eligible_seeds = [
        i for i in range(n)
        if i not in vaccinated_indices
    ]

    for beta in beta_values:

        expected_offspring = []

        for i in eligible_seeds:

            failure = {}

            for j, duration in adjacency[i]:

                if (
                    j != i
                    and j not in vaccinated_indices
                ):

                    p = -np.expm1(
                        -beta * duration
                    )

                    failure[j] = (
                        failure.get(j, 1.0) * (1 - p)
                    )

            expected = sum(
                1 - f
                for f in failure.values()
            )

            expected_offspring.append(expected)

        print(f"\nBeta = {beta}")

        print(
            "Mean expected secondary infections:",
            np.mean(expected_offspring)
        )

        print(
            "Median expected secondary infections:",
            np.median(expected_offspring)
        )

        print(
            "Maximum expected secondary infections:",
            np.max(expected_offspring)
        )

        print(
            "Seeds with zero expected transmission:",
            np.mean(
                np.array(expected_offspring) == 0
            ) * 100,
            "%"
        )

    return network_df


if __name__ == "__main__":

    diagnose_network()