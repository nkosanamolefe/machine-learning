import great_expectations as ge
import pandas as pd
from typing import Tuple, List

def validate_telco_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate Telco Customer Churn dataset using Great Expectations (v1.5.8).
    """
    print("🔍 Starting data validation with Great Expectations...")

    # Convert to GE dataset
    ge_df = ge.get_context()

    # === SCHEMA VALIDATION ===
    print("   📋 Validating schema and required columns...")

    required_columns = [
        "customerID", "gender", "Partner", "Dependents",
        "PhoneService", "InternetService", "Contract",
        "tenure", "MonthlyCharges", "TotalCharges"
    ]

    for col in required_columns:
        ge_df.expect_column_to_exist(col)

    ge_df.expect_column_values_to_not_be_null("customerID")

    # === BUSINESS LOGIC VALIDATION ===
    print("   💼 Validating business logic constraints...")

    ge_df.expect_column_values_to_be_in_set("gender", ["Male", "Female"])
    ge_df.expect_column_values_to_be_in_set("Partner", ["Yes", "No"])
    ge_df.expect_column_values_to_be_in_set("Dependents", ["Yes", "No"])
    ge_df.expect_column_values_to_be_in_set("PhoneService", ["Yes", "No"])
    ge_df.expect_column_values_to_be_in_set("Contract", ["Month-to-month", "One year", "Two year"])
    ge_df.expect_column_values_to_be_in_set("InternetService", ["DSL", "Fiber optic", "No"])

    # === NUMERIC RANGE VALIDATION ===
    print("   📊 Validating numeric ranges and business constraints...")

    ge_df.expect_column_values_to_be_between("tenure", min_value=0, max_value=120)
    ge_df.expect_column_values_to_be_between("MonthlyCharges", min_value=0, max_value=200)
    ge_df.expect_column_values_to_be_between("TotalCharges", min_value=0)

    # === CONSISTENCY CHECKS ===
    print("   🔗 Validating data consistency...")

    ge_df.expect_column_pair_values_A_to_be_greater_than_B(
        column_A="TotalCharges",
        column_B="MonthlyCharges",
        or_equal=True,
        mostly=0.95
    )

    # === RUN VALIDATION SUITE ===
    print("   ⚙️  Running complete validation suite...")
    results = ge_df.validate()

    failed_expectations = [
        r.expectation_config.expectation_type
        for r in results.results
        if not r.success
    ]

    total_checks = len(results.results)
    passed_checks = sum(r.success for r in results.results)
    failed_checks = total_checks - passed_checks

    if results.success:
        print(f"✅ Data validation PASSED: {passed_checks}/{total_checks} checks successful")
    else:
        print(f"❌ Data validation FAILED: {failed_checks}/{total_checks} checks failed")
        print(f"   Failed expectations: {failed_expectations}")

    return results.success, failed_expectations


if __name__ == "__main__":
    # Sample data for testing
    sample_df = pd.DataFrame({
        "customerID": ["C001", "C002", "C003"],
        "gender": ["Male", "Female", "Male"],
        "Partner": ["Yes", "No", "Yes"],
        "Dependents": ["No", "Yes", "No"],
        "PhoneService": ["Yes", "Yes", "No"],
        "InternetService": ["DSL", "Fiber optic", "No"],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "tenure": [12, 24, 6],
        "MonthlyCharges": [65.5, 85.0, 45.0],
        "TotalCharges": [786.0, 2040.0, 270.0]
    })

    success, failed = validate_telco_data(sample_df)
    print(f"\nValidation Success: {success}")
    print(f"Failed Expectations: {failed}")
