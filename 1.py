#!/usr/bin/env python3
"""
Student Enrollment Analysis Script

This script analyzes student enrollment data from a CSV file and answers
various questions about class distributions, patterns, and statistics.

Usage: python student_analysis.py [csv_file_path]
Default CSV file: q-fastapi.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import json
import sys
import os
from typing import Dict, List, Any


def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """
    Load the CSV file and perform data cleaning as specified in the plan.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Cleaned pandas DataFrame
    """
    try:
        # Load the CSV file
        df = pd.read_csv(file_path, sep=',', header=0, encoding='utf-8')
        print(f"Loaded data with {len(df)} rows and {len(df.columns)} columns")
        
        # Ensure required columns exist
        required_columns = ['studentId', 'class']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Missing required columns. Expected: {required_columns}, Found: {df.columns.tolist()}")
        
        # Clean studentId column - convert to int, drop NaN rows
        df['studentId'] = pd.to_numeric(df['studentId'], errors='coerce')
        initial_rows = len(df)
        df = df.dropna(subset=['studentId'])
        df['studentId'] = df['studentId'].astype('int64')
        print(f"Dropped {initial_rows - len(df)} rows with invalid studentId")
        
        # Clean class column - ensure string type
        df['class'] = df['class'].astype(str)
        
        # Remove rows with missing class data
        initial_rows = len(df)
        df = df.dropna(subset=['class'])
        df = df[df['class'] != 'nan']  # Remove string 'nan' values
        print(f"Dropped {initial_rows - len(df)} rows with missing class data")
        
        # Remove completely duplicate rows
        initial_rows = len(df)
        df = df.drop_duplicates()
        print(f"Dropped {initial_rows - len(df)} duplicate rows")
        
        print(f"Final cleaned data: {len(df)} rows")
        return df
        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)


def q1_unique_classes_count(df: pd.DataFrame) -> int:
    """How many unique classes are there?"""
    return df['class'].nunique()


def q2_class_highest_students(df: pd.DataFrame) -> str:
    """Which class has the highest number of students?"""
    return df.groupby('class').size().idxmax()


def q3_average_studentid_per_class(df: pd.DataFrame) -> Dict[str, float]:
    """What is the average studentId for each class?"""
    return df.groupby('class')['studentId'].mean().to_dict()


def q4_students_per_class_barchart(df: pd.DataFrame) -> str:
    """Draw a bar chart showing the number of students per class."""
    student_counts = df['class'].value_counts().sort_index()
    
    plt.figure(figsize=(12, 6))
    student_counts.plot(kind='bar')
    plt.title('Number of Students Per Class')
    plt.xlabel('Class')
    plt.ylabel('Number of Students')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save to base64 string
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    
    base64_string = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{base64_string}"


def q5_classes_with_one_student(df: pd.DataFrame) -> List[str]:
    """Which classes have only one student enrolled?"""
    student_counts = df.groupby('class').size()
    return student_counts[student_counts == 1].index.tolist()


def q6_student_counts_distribution(df: pd.DataFrame) -> Dict[str, float]:
    """What is the distribution of student counts across classes?"""
    return df.groupby('class').size().describe().to_dict()


def q7_class_pattern_counts(df: pd.DataFrame) -> Dict[str, int]:
    """Is there a class code pattern (e.g., ending in E or starting with 8)? List all such patterns with counts."""
    ending_e_students = df[df['class'].astype(str).str.endswith('E', na=False)].shape[0]
    starting_8_students = df[df['class'].astype(str).str.startswith('8', na=False)].shape[0]
    
    return {
        'students_in_classes_ending_in_E': ending_e_students,
        'students_in_classes_starting_with_8': starting_8_students
    }


def q8_percentage_students_class_starts_with_8(df: pd.DataFrame) -> float:
    """What percentage of students belong to classes that start with the digit '8'?"""
    total_students = len(df)
    students_starting_8 = df[df['class'].astype(str).str.startswith('8', na=False)].shape[0]
    percentage = (students_starting_8 / total_students) * 100 if total_students > 0 else 0
    return percentage


def run_analysis(df: pd.DataFrame) -> List[str]:
    """
    Run all analysis tasks and return results in the specified format.
    
    Returns:
        List of string results for each question
    """
    results = []
    
    print("Running analysis tasks...")
    
    # Q1: Unique classes count
    print("1. Counting unique classes...")
    result1 = q1_unique_classes_count(df)
    results.append(str(result1))
    print(f"   Result: {result1}")
    
    # Q2: Class with highest students
    print("2. Finding class with highest number of students...")
    result2 = q2_class_highest_students(df)
    results.append(result2)
    print(f"   Result: {result2}")
    
    # Q3: Average studentId per class
    print("3. Calculating average studentId per class...")
    result3 = q3_average_studentid_per_class(df)
    results.append(json.dumps(result3))
    print(f"   Result: {result3}")
    
    # Q4: Bar chart
    print("4. Generating bar chart...")
    result4 = q4_students_per_class_barchart(df)
    results.append(result4)
    print("   Result: Bar chart generated (base64 encoded)")
    
    # Q5: Classes with one student
    print("5. Finding classes with only one student...")
    result5 = q5_classes_with_one_student(df)
    results.append(json.dumps(result5))
    print(f"   Result: {result5}")
    
    # Q6: Distribution statistics
    print("6. Calculating student count distribution...")
    result6 = q6_student_counts_distribution(df)
    results.append(json.dumps(result6))
    print(f"   Result: {result6}")
    
    # Q7: Pattern analysis
    print("7. Analyzing class code patterns...")
    result7 = q7_class_pattern_counts(df)
    results.append(json.dumps(result7))
    print(f"   Result: {result7}")
    
    # Q8: Percentage calculation
    print("8. Calculating percentage of students in classes starting with '8'...")
    result8 = q8_percentage_students_class_starts_with_8(df)
    results.append(str(result8))
    print(f"   Result: {result8}%")
    
    return results


def save_results(results: List[str], output_file: str = "analysis_results.json"):
    """Save results to a JSON file."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")


def main():
    """Main function to orchestrate the analysis."""
    # Get CSV file path from command line or use default
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "q-fastapi.csv"
    
    print(f"Student Enrollment Analysis")
    print(f"{'='*50}")
    print(f"CSV file: {csv_file}")
    
    # Check if file exists
    if not os.path.exists(csv_file):
        print(f"Error: File '{csv_file}' not found.")
        print("Please ensure the CSV file exists or provide the correct path as an argument.")
        sys.exit(1)
    
    # Load and clean data
    print(f"\nLoading and cleaning data...")
    student_data = load_and_clean_data(csv_file)
    
    # Display basic info
    print(f"\nData Overview:")
    print(f"Total students (rows): {len(student_data)}")
    print(f"Unique classes: {student_data['class'].nunique()}")
    print(f"Student ID range: {student_data['studentId'].min()} - {student_data['studentId'].max()}")
    print(f"\nFirst few rows:")
    print(student_data.head())
    
    # Run analysis
    print(f"\n{'='*50}")
    results = run_analysis(student_data)
    
    # Save results
    save_results(results)
    
    # Display final results summary
    print(f"\n{'='*50}")
    print("FINAL RESULTS SUMMARY:")
    print("=" * 50)
    
    questions = [
        "How many unique classes are there?",
        "Which class has the highest number of students?", 
        "What is the average studentId for each class?",
        "Draw a bar chart showing the number of students per class.",
        "Which classes have only one student enrolled?",
        "What is the distribution of student counts across classes?",
        "Is there a class code pattern (e.g., ending in E or starting with 8)? List all such patterns with counts.",
        "What percentage of students belong to classes that start with the digit '8'?"
    ]
    
    for i, (question, result) in enumerate(zip(questions, results), 1):
        print(f"\nQ{i}: {question}")
        if result.startswith("data:image/png;base64,"):
            print("   Answer: [Bar chart generated - see chart output]")
        else:
            # Pretty print JSON if it's JSON
            try:
                parsed = json.loads(result)
                if isinstance(parsed, dict):
                    print("   Answer:")
                    for key, value in parsed.items():
                        print(f"     {key}: {value}")
                elif isinstance(parsed, list):
                    print(f"   Answer: {', '.join(map(str, parsed))}")
                else:
                    print(f"   Answer: {result}")
            except (json.JSONDecodeError, TypeError):
                print(f"   Answer: {result}")


if __name__ == "__main__":
    main()