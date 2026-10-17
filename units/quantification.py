"""
Droplet Digital PCR Quantification Module

This module provides functions for:
- Newton-Raphson iteration for concentration calculation
- Droplet volume estimation from area measurements
- Target concentration quantification from droplet classification
"""

import math
import pandas as pd
import openpyxl as op
import time

def func(pos_volumes, neg_volumes, concentration):
    """
    Function for Newton-Raphson iteration to solve concentration.

    Args:
        pos_volumes (list): Volumes of positive droplets (μL)
        neg_volumes (list): Volumes of negative droplets (μL)
        concentration (float): Current concentration estimate (copies/μL)

    Returns:
        float: Function value at current concentration
    """
    total_volume = sum(pos_volumes) + sum(neg_volumes)
    sigma = 0.0

    for vol in pos_volumes:
            sigma += vol / (1 - math.exp(-vol * concentration))

    return total_volume - sigma


def func_prime(pos_volumes, neg_volumes, concentration):
    """
    Derivative of the Newton-Raphson function.

    Args:
        pos_volumes (list): Volumes of positive droplets (μL)
        neg_volumes (list): Volumes of negative droplets (μL)
        concentration (float): Current concentration estimate (copies/μL)

    Returns:
        float: Derivative value at current concentration
    """
    sigma = 0.0
    for vol in pos_volumes:
        numerator = (vol ** 2) * math.exp(-vol * concentration)
        denominator = (1 - math.exp(-vol * concentration)) ** 2
        sigma += numerator / denominator
    return sigma


def get_concentration(pos_volumes, neg_volumes):
    """
    Calculate concentration using Newton-Raphson iteration.

    Args:
        pos_volumes (list): Volumes of positive droplets (μL)
        neg_volumes (list): Volumes of negative droplets (μL)

    Returns:
        float: Calculated concentration (copies/μL)
    """
    # Initial concentration estimate
    pos_count = float(len(pos_volumes))
    neg_count = float(len(neg_volumes))
    total_count = pos_count + neg_count

    if total_count <= 0:
        return 0.0

    # No positive droplets means no detected copies.
    if pos_count <= 0:
        return 0.0

    total_volume = sum(pos_volumes) + sum(neg_volumes)
    avg_volume = total_volume / total_count

    if avg_volume <= 0:
        return 0.0

    if pos_count >= total_count:
        return 0.0

    concentration = -math.log(1 - pos_count / total_count) / avg_volume

    # Newton-Raphson iteration
    MAX_ITERATIONS = 1000
    TOLERANCE = 1e-9

    for iteration in range(MAX_ITERATIONS):
        print(f"Iteration {iteration + 1}: c = {concentration:.4f}")
        derivative = func_prime(pos_volumes, neg_volumes, concentration)
        if derivative == 0:
            break

        new_concentration = concentration - (
                func(pos_volumes, neg_volumes, concentration) / derivative
        )

        if concentration == 0:
            concentration = max(new_concentration, 0.0)
            break

        if abs((new_concentration - concentration) / concentration) < TOLERANCE:
            break

        concentration = max(new_concentration, 0.0)

    print(f"Final concentration: {concentration} copies/μL")
    return concentration


def calculate_volume(area_px, scale_bar, critical_diameter):
    """
    Calculate droplet volume from area measurement.

    Args:
        area_px (float): Droplet area in pixels
        scale_bar (float): Micrometers per pixel conversion factor
        critical_diameter (float): Threshold diameter for volume calculation mode

    Returns:
        float: Calculated volume (μL)
    """
    area_um2 = area_px * (scale_bar ** 2)
    d_eq = math.sqrt(4 * area_um2 / math.pi)  # equivalent circle diameter (μm)

    if d_eq > critical_diameter:
        # Disk-shaped droplet confined by chip walls (height = 62 μm)
        # volume (μm³) × 1e-9 → μL
        return area_um2 * 62 * 1e-9
    else:
        # Polynomial fit: true 3-D diameter D (μm) from equivalent 2-D diameter d_eq
        D = (3.085817e-06 * d_eq ** 3
             - 2.368324e-03 * d_eq ** 2
             + 1.070568e+00 * d_eq)
        # Sphere volume (μm³) × 1e-9 → μL
        return (4 / 3) * math.pi * (D / 2) ** 3 * 1e-9


def main():
    """Simulate BINARY droplet quantification process."""
    # Simulation parameters
    NUM_DROPLETS = 5000
    SCALE_BAR = 1.72
    D_CR = 170
    SAVE_PATH = "simulated_quantification_binary.xlsx"

    # Generate simulated droplet areas
    np.random.seed(42)
    areas = np.random.lognormal(mean=5, sigma=0.5, size=NUM_DROPLETS)

    # 1. 修改为二分类随机生成 (0=Neg, 1=Pos)
    print("Generating binary simulation data...")
    classes = np.random.choice([0, 1], size=NUM_DROPLETS, p=[0.7, 0.3])

    # 2. 分别计算体积
    vols_neg = [calculate_volume(a, SCALE_BAR, D_CR) for a, c in zip(areas, classes) if c == 0]
    vols_pos = [calculate_volume(a, SCALE_BAR, D_CR) for a, c in zip(areas, classes) if c == 1]

    # Calculate statistics
    N_total = len(vols_neg) + len(vols_pos)

    # 3. 直接计算单靶标浓度
    print("\n--- Single Target Quantification ---")
    start = time.time()

    # 核心：直接传入阳性和阴性体积列表
    concentration = get_concentration(vols_pos, vols_neg)

    elapsed = time.time() - start
    print(f"Calculation time: {elapsed:.6f} seconds")
    print(f"Result: {concentration:.4f} copies/μL")

    # Save results
    workbook = op.Workbook()
    sheet = workbook.active
    sheet.append(["Parameter", "Value"])
    sheet.append(["Concentration (copies/μL)", concentration])
    sheet.append(["Positive Droplets", len(vols_pos)])
    sheet.append(["Negative Droplets", len(vols_neg)])
    sheet.append(["Total Droplets", N_total])
    workbook.save(SAVE_PATH)
    print(f"Simulation saved to {SAVE_PATH}")


if __name__ == "__main__":
    import numpy as np  # Required for simulation
    main()
