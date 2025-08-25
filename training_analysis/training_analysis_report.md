# Training Circuit Analysis Report

## 📊 Analysis Summary

**Fixed Optimization Sequence**: `balance; rewrite; refactor; balance; rewrite`
**Total Circuits Analyzed**: 552
**Analysis Date**: 2025-07-15 22:54:09

## 📈 Key Statistics

- **Mean Area Reduction**: 15.25%
- **Median Area Reduction**: 16.00%
- **Standard Deviation**: 13.05%
- **Range**: 0.00% - 74.70%

## 🎯 Distribution Analysis

- **Zero Reduction**: 104 circuits (18.8%)
- **Low Reduction (1-10%)**: 129 circuits (23.4%)
- **Medium Reduction (10-30%)**: 244 circuits (44.2%)
- **High Reduction (≥30%)**: 75 circuits (13.6%)

## 🏆 Top 10 Performers

1. **ex50.aag** - 74.70% reduction (83 → 21)
2. **i8.aag** - 63.23% reduction (3310 → 1217)
3. **e64.aag** - 61.14% reduction (1436 → 558)
4. **ex35.aag** - 50.00% reduction (34 → 17)
5. **ex37.aag** - 48.34% reduction (1024 → 529)
6. **ex32.aag** - 48.33% reduction (120 → 62)
7. **ex11.aag** - 47.92% reduction (48 → 25)
8. **ex52.aag** - 47.62% reduction (42 → 22)
9. **ex49.aag** - 46.67% reduction (225 → 120)
10. **parity.aag** - 46.43% reduction (84 → 45)


## 📉 Bottom 10 Performers

1. **bar.aag** - 0.00% reduction (2952 → 2952)
2. **dec.aag** - 0.00% reduction (304 → 304)
3. **int2float.aag** - 0.00% reduction (200 → 200)
4. **c17.aag** - 0.00% reduction (6 → 6)
5. **c432.aag** - 0.00% reduction (122 → 122)
6. **c6288.aag** - 0.00% reduction (1870 → 1870)
7. **9sym.aag** - 0.00% reduction (54 → 54)
8. **C17.aag** - 0.00% reduction (6 → 6)
9. **C432.aag** - 0.00% reduction (122 → 122)
10. **C6288.aag** - 0.00% reduction (1870 → 1870)


## 🔍 Outlier Analysis

**Outliers Identified**: 1 circuits

### High Outliers (Exceptional Performance)


### Low Outliers (Poor Performance)
1. **ex50.aag** - 74.70% reduction


## 📊 Statistical Details

- **Q1 (25th percentile)**: 2.60%
- **Q3 (75th percentile)**: 23.51%
- **IQR**: 20.90%

## 🎯 Insights

1. **Circuit Diversity**: The training set shows diverse optimization potential
2. **Optimization Effectiveness**: 75 circuits show significant improvement
3. **Already Optimized**: 104 circuits are already near-optimal
4. **Outlier Patterns**: Outliers may represent special circuit types or optimization challenges

## 📋 Recommendations

1. **Focus on High-Performing Circuits**: Prioritize circuits with >30% reduction for RL training
2. **Analyze Zero-Reduction Circuits**: Investigate why some circuits don't benefit from optimization
3. **Study Outliers**: Understand what makes exceptional and poor performers different
4. **Circuit Filtering**: Consider filtering out circuits with <5% reduction for training efficiency
