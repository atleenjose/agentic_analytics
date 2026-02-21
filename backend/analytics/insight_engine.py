def generate_insight(summary: dict) -> str:
    insights = []

    avg_cost = summary.get("avg_cost", 0)
    max_cost = summary.get("max_cost", 0)
    min_cost = summary.get("min_cost", 0)
    outliers = summary.get("outlier_count", 0)
    tiers = summary.get("tier_distribution", [])

    # Guard: no data
    if summary.get("total_rows", 0) == 0:
        return "Dataset is empty. No insights can be generated."

    # 1. Cost dispersion insight
    if avg_cost > 0 and max_cost > avg_cost * 2:
        ratio = round(max_cost / avg_cost, 1)
        insights.append(
            f"Cost distribution exhibits right-tail skew. Maximum cost (${max_cost}) "
            f"is {ratio}x the average, indicating burst usage behavior driving cost spikes."
        )

    # 2. Outlier risk
    if outliers > 0:
        insights.append(
            f"{outliers} statistical outliers were identified using the IQR method. "
            "Even low-frequency anomalies can materially impact monthly cost volatility."
        )

    # 3. Tier comparison
    if tiers:
        tier_costs = {t["user_tier"]: t["avg_cost"] for t in tiers}
        highest_tier = max(tier_costs, key=tier_costs.get)

        insights.append(
            f"Tier '{highest_tier}' users demonstrate the highest average cost "
            f"(${round(tier_costs[highest_tier], 4)}), suggesting greater computational intensity."
        )

        # 4. Cost concentration
        total_costs = [t["total_cost"] for t in tiers]
        total_sum = sum(total_costs)

        if total_sum > 0:
            concentration_ratio = max(total_costs) / total_sum
            if concentration_ratio > 0.35:
                insights.append(
                    "Cost contribution is moderately concentrated within a single tier, "
                    "creating sensitivity to behavioral shifts in that segment."
                )

    # 5. Margin monitoring
    insights.append(
        f"Average cost per conversation is ${round(avg_cost,4)}. "
        "Sustained monitoring of token utilization and session duration "
        "is required to protect gross margins at scale."
    )

    return "\n\n".join(f"{i+1}. {insights[i]}" for i in range(len(insights)))