# run_complete_system.py
#!/usr/bin/env python3
"""
Complete system execution + analysis + plotting
"""

import sys
import os

# 1. Run main experiment
print("🔬 Step 1: Running experiments")
os.system("python -m ci.experiments.run_orchestrator_eval")

# 2. Analyze results
print("\n📊 Step 2: Analyzing results")
os.system("python -m ci.experiments.analyze_and_plot")

# 3. Generate report
print("\n📋 Step 3: Generating final report")
os.system("python -m ci.experiments.generate_report")

print("\n🎉 System execution completed!")
print("📁 Results in results/ folder")
print("📈 Plots in results/figures/")