#!/usr/bin/env python3
"""Generador simple de transacciones de venta para pruebas.

Genera un CSV con columnas mínimas requeridas por `service_friendly.py`:
- CustomerID, InvoiceDate, Quantity, UnitPrice, InvoiceNo, Description

Uso:
    python scripts/generate_sample_sales.py --rows 1000 --out data/sample_sales.csv

El script intenta crear transacciones realistas distribuidas entre varios clientes
y productos para permitir pruebas de segmentación.
"""
from __future__ import annotations

import argparse
import csv
import random
from datetime import datetime, timedelta
from pathlib import Path


PRODUCTS = [
    ("Red T-shirt", 15.0, 35.0),
    ("Leather Wallet", 40.0, 120.0),
    ("Notebook", 3.0, 20.0),
    ("Coffee Mug", 5.0, 25.0),
    ("Shoes", 30.0, 150.0),
    ("Smartwatch", 120.0, 450.0),
    ("Pen Set", 2.0, 15.0),
    ("Hat", 8.0, 45.0),
    ("Sunglasses", 20.0, 200.0),
    ("Backpack", 25.0, 180.0),
]


def random_date(start: datetime, end: datetime) -> datetime:
    span = (end - start).total_seconds()
    offset = random.random() * span
    return start + timedelta(seconds=offset)


def generate_transactions(rows: int, out_path: Path, start_date: datetime, end_date: datetime, unique_customers: int | None = None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if unique_customers is None:
        # elegir un número de clientes razonable para distribuir compras
        unique_customers = max(50, min(rows // 2, 500))

    customer_ids = [f"CUST_{i:04d}" for i in range(1, unique_customers + 1)]

    invoice_no = 1000
    with out_path.open("w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["CustomerID", "InvoiceDate", "Quantity", "UnitPrice", "InvoiceNo", "Description"])

        for i in range(rows):
            cust = random.choice(customer_ids)
            dt = random_date(start_date, end_date)
            # algunas filas representan compras grandes, otras pequeñas
            product, lo, hi = random.choice(PRODUCTS)
            # unit price dentro del rango, con sesgo hacia valores bajos
            unit_price = round(random.uniform(lo, hi) * random.choice([0.9, 1.0, 1.0, 1.2]), 2)
            # quantity con distribución: la mayoría 1-3, raros >5
            q = random.choices([1, 2, 3, 4, 5, 8, 10], weights=[40, 30, 12, 8, 6, 3, 1], k=1)[0]
            writer.writerow([cust, dt.strftime("%Y-%m-%d"), q, unit_price, invoice_no, product])
            invoice_no += 1


def main():
    p = argparse.ArgumentParser(description="Generar CSV de transacciones de ejemplo")
    p.add_argument("--rows", type=int, default=1000, help="Número de transacciones (filas) a generar")
    p.add_argument("--out", type=str, default="data/sample_sales.csv", help="Ruta de salida para el CSV")
    p.add_argument("--days", type=int, default=30, help="Periodo en días hacia atrás desde hoy (por defecto 30)")
    p.add_argument("--customers", type=int, default=0, help="Número de clientes únicos. 0 = auto")
    args = p.parse_args()

    end = datetime.today()
    start = end - timedelta(days=args.days)
    out_path = Path(args.out)

    customers = args.customers if args.customers > 0 else None
    generate_transactions(args.rows, out_path, start, end, unique_customers=customers)
    print(f"Generado {args.rows} transacciones en {out_path} (clientes únicos: {customers or 'auto'})")


if __name__ == "__main__":
    main()
