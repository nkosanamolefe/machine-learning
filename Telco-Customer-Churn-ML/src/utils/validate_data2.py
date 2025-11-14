import logging
from typing import Tuple, List, Dict, Any
import pandas as pd
import great_expectations as gx
from great_expectations.core.batch import Batch
from great_expectations.core.expectation_suite import ExpectationSuite
from great_expectations.validator.validator import Validator
from great_expectations.execution_engine.pandas_execution_engine import PandasExecutionEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration constants
VALIDATION_CONFIG = {
    "required_columns": [
        "customerID", "gender", "Partner", "Dependents", 
        "PhoneService", "InternetService", "Contract", 
        "tenure", "MonthlyCharges", "TotalCharges"
    ],
    "categorical_constraints": {
        "gender": ["Male", "Female"],
        "Partner": ["Yes", "No"],
        "Dependents": ["Yes", "No"],
        "PhoneService": ["Yes", "No"],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "InternetService": ["DSL", "Fiber optic", "No"]
    },
    "numeric_ranges": {
        "tenure": {"min": 0, "max": 120},
        "MonthlyCharges": {"min": 0, "max": 200},
        "TotalCharges": {"min": 0, "max": 10000}
    },
    "non_nullable_columns": ["customerID", "gender", "tenure", "MonthlyCharges", "TotalCharges", "Contract"]
}

def validate_telco_data(df: pd.DataFrame) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Validate telco customer data using Great Expectations.
    
    Args:
        df: Pandas DataFrame containing telco customer data
        
    Returns:
        Tuple of (validation_success, failed_expectations, detailed_results)
    """
    logger.info("🔍 Starting data validation with Great Expectations...")
    
    # Validate input
    if df is None or df.empty:
        logger.error("❌ Input DataFrame is None or empty")
        return False, ["empty_dataframe"], {"total_checks": 0, "passed": 0, "failed": 0, "row_count": 0}
    
    logger.info(f"   📊 Processing {len(df)} rows")
    
    try:
        # Initialize Great Expectations context (ephemeral, no persistence)
        context = gx.get_context()
        
        # Create execution engine and batch
        execution_engine = PandasExecutionEngine()
        batch = Batch(data=df)
        
        # Create expectation suite
        expectation_suite = ExpectationSuite(name="telco_validation")
        
        # Create validator with batch and context
        validator = Validator(
            execution_engine=execution_engine,
            batches=[batch],
            expectation_suite=expectation_suite,
            data_context=context
        )
        
        # === SCHEMA VALIDATION ===
        logger.info("   📋 Validating schema and required columns...")
        for col in VALIDATION_CONFIG["required_columns"]:
            validator.expect_column_to_exist(col)
        
        # === NULL VALUE VALIDATION ===
        logger.info("   🔒 Validating null constraints...")
        for col in VALIDATION_CONFIG["non_nullable_columns"]:
            validator.expect_column_values_to_not_be_null(col)
        
        # === CATEGORICAL VALIDATION ===
        logger.info("   💼 Validating categorical values...")
        for col, allowed_values in VALIDATION_CONFIG["categorical_constraints"].items():
            validator.expect_column_values_to_be_in_set(col, allowed_values)
        
        # === NUMERIC RANGE VALIDATION ===
        logger.info("   📈 Validating numeric ranges...")
        for col in VALIDATION_CONFIG["numeric_ranges"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        for col, bounds in VALIDATION_CONFIG["numeric_ranges"].items():
            validator.expect_column_values_to_be_between(
                col,
                min_value=bounds["min"],
                max_value=bounds["max"]
            )
        
        # === DATA CONSISTENCY CHECK ===
        logger.info("   🔗 Validating data consistency...")
        validator.expect_column_pair_values_A_to_be_greater_than_B(
            column_A="TotalCharges",
            column_B="MonthlyCharges",
            or_equal=True,
            mostly=0.95
        )
        
        # === CARDINALITY CHECK ===
        logger.info("   📏 Validating data cardinality...")
        validator.expect_table_row_count_to_be_between(min_value=1, max_value=1000000)
        
        # Get results
        results = validator.validate()
        
        # Process results
        failed_expectations = []
        failed_details = []
        
        for result in results.results:
            if not result.success:
                expectation_type = result["expectation_config"]["expectation_context"]
                failed_expectations.append(expectation_type)
                failed_details.append({
                    "expectation": expectation_type,
                    "kwargs": result.expectation_config.get("kwargs", {}),
                    "message": result.result.get("result_message", "No details available")
                })
        
        total_checks = len(results.results)
        passed_checks = sum(1 for r in results.results if r.success)
        failed_checks = total_checks - passed_checks
        
        # Log results
        if results.success:
            logger.info(f"✅ Data validation PASSED: {passed_checks}/{total_checks} checks successful")
        else:
            logger.warning(f"❌ Data validation FAILED: {failed_checks}/{total_checks} checks failed")
            for detail in failed_details:
                logger.warning(f"   ✗ {detail['expectation']}: {detail['message']}")
        
        detailed_results = {
            "total_checks": total_checks,
            "passed": passed_checks,
            "failed": failed_checks,
            "row_count": len(df),
            "failed_details": failed_details
        }
        
        return results.success, failed_expectations, detailed_results
        
    except Exception as e:
        logger.error(f"❌ Validation error: {str(e)}", exc_info=True)
        return False, ["validation_error"], {"error": str(e), "row_count": len(df) if df is not None else 0}


# Example usage
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
    
    success, failed, details = validate_telco_data(sample_df)
    print(f"\nValidation Success: {success}")
    print(f"Failed Expectations: {failed}")
    print(f"Details: {details}")