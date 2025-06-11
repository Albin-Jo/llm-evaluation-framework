import logging
import math
from io import BytesIO
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
from fastapi import HTTPException
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import PageBreak
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from starlette import status

logger = logging.getLogger(__name__)
# Maximum percentage change to avoid infinity values
MAX_PERCENTAGE_CHANGE = 999.9


class StatisticsUtils:
    """Enhanced utility class for statistical calculations."""

    # Effect size interpretation thresholds
    EFFECT_SIZE_THRESHOLDS = {
        "negligible": 0.2,
        "small": 0.5,
        "medium": 0.8
    }

    @staticmethod
    def safe_percentage_change(new_value: Optional[float], old_value: Optional[float], cap_extreme: bool = True) -> \
    Optional[float]:
        """Calculate percentage change with safe handling of edge cases and None values."""
        if new_value is None or old_value is None:
            logger.debug(f"Cannot calculate percentage change: new_value={new_value}, old_value={old_value}")
            return None

        if new_value == old_value:
            return 0.0

        if abs(old_value) > 1e-10:  # Normal case
            percentage = ((new_value - old_value) / old_value) * 100
        else:  # Division by zero or very small numbers
            if cap_extreme:
                if new_value > old_value:
                    percentage = MAX_PERCENTAGE_CHANGE
                elif new_value < old_value:
                    percentage = -MAX_PERCENTAGE_CHANGE
                else:
                    percentage = 0.0
            else:
                percentage = float('inf') if new_value > old_value else float('-inf')

        # Cap extreme values if requested
        if cap_extreme and percentage is not None:
            percentage = max(-MAX_PERCENTAGE_CHANGE, min(MAX_PERCENTAGE_CHANGE, percentage))

        return percentage

    @staticmethod
    def safe_absolute_difference(new_value: Optional[float], old_value: Optional[float]) -> Optional[float]:
        """Calculate absolute difference with safe handling of None values."""
        if new_value is None or old_value is None:
            return None
        return new_value - old_value

    @staticmethod
    def normalize_metric_values(values: List[float], source_scale: Tuple[float, float],
                                target_scale: Tuple[float, float] = (0, 1)) -> List[float]:
        """Normalize metric values from source scale to target scale."""
        if not values:
            return []

        source_min, source_max = source_scale
        target_min, target_max = target_scale

        # Handle case where source scale is a single point
        if source_max == source_min:
            return [target_min] * len(values)

        normalized = []
        for value in values:
            # Normalize to 0-1 first
            normalized_01 = (value - source_min) / (source_max - source_min)
            # Scale to target range
            scaled_value = target_min + normalized_01 * (target_max - target_min)
            normalized.append(scaled_value)

        return normalized

    @staticmethod
    def apply_multiple_comparison_correction(p_values: List[Optional[float]],
                                             method: str = 'bonferroni') -> List[Optional[float]]:
        """Apply multiple comparison correction to p-values."""
        if not p_values or not any(p is not None for p in p_values):
            return p_values

        # Filter out None values for processing
        valid_p_values = [(i, p) for i, p in enumerate(p_values) if p is not None]

        if not valid_p_values:
            return p_values

        corrected = [None] * len(p_values)

        if method == 'bonferroni':
            # Bonferroni correction: multiply by number of tests
            n_tests = len(valid_p_values)
            for i, p in valid_p_values:
                corrected[i] = min(p * n_tests, 1.0)

        elif method == 'fdr':  # Benjamini-Hochberg
            # Sort p-values
            sorted_p = sorted(valid_p_values, key=lambda x: x[1])
            n_tests = len(sorted_p)

            for rank, (original_idx, p_value) in enumerate(sorted_p, 1):
                # FDR correction: p * n / rank
                corrected_p = min(p_value * n_tests / rank, 1.0)
                corrected[original_idx] = corrected_p

        return corrected

    @staticmethod
    def calculate_effect_size(values_a: List[float], values_b: List[float]) -> Optional[float]:
        """Calculate Cohen's d effect size."""
        if not values_a or not values_b or len(values_a) < 2 or len(values_b) < 2:
            return None

        try:
            mean_a = np.mean(values_a)
            mean_b = np.mean(values_b)

            # Pooled standard deviation
            var_a = np.var(values_a, ddof=1)
            var_b = np.var(values_b, ddof=1)
            pooled_std = math.sqrt(((len(values_a) - 1) * var_a + (len(values_b) - 1) * var_b) /
                                   (len(values_a) + len(values_b) - 2))

            if pooled_std == 0:
                return 0.0

            cohens_d = (mean_b - mean_a) / pooled_std
            return cohens_d

        except Exception as e:
            logger.warning(f"Failed to calculate effect size: {e}")
            return None

    @classmethod
    def interpret_effect_size(cls, effect_size: float) -> str:
        """Interpret Cohen's d effect size using class thresholds."""
        if effect_size is None:
            return "unknown"

        abs_effect = abs(effect_size)
        if abs_effect < cls.EFFECT_SIZE_THRESHOLDS["negligible"]:
            return "negligible"
        elif abs_effect < cls.EFFECT_SIZE_THRESHOLDS["small"]:
            return "small"
        elif abs_effect < cls.EFFECT_SIZE_THRESHOLDS["medium"]:
            return "medium"
        else:
            return "large"

    @staticmethod
    def calculate_confidence_interval(values: List[float], confidence_level: float = 0.95) -> Optional[
        Tuple[float, float]]:
        """Calculate confidence interval for a list of values."""
        if not values or len(values) < 2:
            return None

        try:
            from scipy import stats as scipy_stats

            mean = np.mean(values)
            std_err = scipy_stats.sem(values)

            # Calculate t-critical value
            df = len(values) - 1
            alpha = 1 - confidence_level
            t_critical = scipy_stats.t.ppf(1 - alpha / 2, df)

            # Calculate margin of error
            margin_of_error = t_critical * std_err

            return (mean - margin_of_error, mean + margin_of_error)

        except Exception as e:
            logger.warning(f"Failed to calculate confidence interval: {e}")
            return None

    @staticmethod
    def calculate_power_analysis(effect_size: float, sample_size: int, alpha: float = 0.05) -> Optional[float]:
        """Calculate statistical power for a given effect size and sample size."""
        try:
            from scipy import stats as scipy_stats

            # Calculate critical t-value
            df = (sample_size - 1) * 2  # Assuming equal sample sizes
            t_critical = scipy_stats.t.ppf(1 - alpha / 2, df)

            # Calculate non-centrality parameter
            ncp = effect_size * math.sqrt(sample_size / 2)

            # Calculate power using non-central t-distribution
            power = 1 - scipy_stats.nct.cdf(t_critical, df, ncp) + scipy_stats.nct.cdf(-t_critical, df, ncp)

            return power

        except Exception as e:
            logger.warning(f"Failed to calculate statistical power: {e}")
            return None

    @staticmethod
    def detect_outliers(values: List[float], method: str = 'iqr') -> List[int]:
        """Detect outliers in a list of values using IQR or Z-score method."""
        if not values or len(values) < 4:
            return []

        try:
            if method == 'iqr':
                # Interquartile Range method
                q1 = np.percentile(values, 25)
                q3 = np.percentile(values, 75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                outliers = [i for i, value in enumerate(values)
                            if value < lower_bound or value > upper_bound]

            elif method == 'zscore':
                # Z-score method
                mean = np.mean(values)
                std = np.std(values)
                z_scores = [(value - mean) / std for value in values]

                outliers = [i for i, z_score in enumerate(z_scores)
                            if abs(z_score) > 2.5]  # 2.5 standard deviations
            else:
                outliers = []

            return outliers

        except Exception as e:
            logger.warning(f"Failed to detect outliers: {e}")
            return []


class PDFReportGenerator:
    """Generate PDF reports for comparisons."""

    def __init__(self):
        try:
            self.letter = letter
            self.A4 = A4
            self.getSampleStyleSheet = getSampleStyleSheet
            self.ParagraphStyle = ParagraphStyle
            self.inch = inch
            self.SimpleDocTemplate = SimpleDocTemplate
            self.Paragraph = Paragraph
            self.Spacer = Spacer
            self.Table = Table
            self.TableStyle = TableStyle
            self.colors = colors
            self.PageBreak = PageBreak

            self.available = True
            logger.info("PDF generation libraries loaded successfully")

        except ImportError as e:
            logger.warning(f"PDF generation not available: {e}")
            self.available = False

    def generate_comparison_pdf(self, comparison_data: Dict[str, Any]) -> BytesIO:
        """Generate PDF report from comparison data."""
        if not self.available:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="PDF generation not available. Please install reportlab."
            )

        buffer = BytesIO()
        doc = self.SimpleDocTemplate(buffer, pagesize=self.A4)
        styles = self.getSampleStyleSheet()
        story = []

        # Title
        title_style = self.ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=1  # Center alignment
        )
        story.append(self.Paragraph(comparison_data["report_title"], title_style))
        story.append(self.Spacer(1, 12))

        # Metadata
        story.append(self.Paragraph(f"<b>Generated:</b> {comparison_data['generated_at']}", styles['Normal']))
        story.append(self.Spacer(1, 12))

        # Evaluations being compared
        story.append(self.Paragraph("<b>Evaluations Compared:</b>", styles['Heading2']))
        eval_a = comparison_data["evaluations"]["evaluation_a"]
        eval_b = comparison_data["evaluations"]["evaluation_b"]

        story.append(
            self.Paragraph(f"<b>Evaluation A:</b> {eval_a['name']} (Method: {eval_a['method']})", styles['Normal']))
        story.append(
            self.Paragraph(f"<b>Evaluation B:</b> {eval_b['name']} (Method: {eval_b['method']})", styles['Normal']))
        story.append(self.Spacer(1, 20))

        # Executive Summary
        if comparison_data.get("summary"):
            story.append(self.Paragraph("<b>Executive Summary</b>", styles['Heading2']))
            summary = comparison_data["summary"]

            story.append(self.Paragraph(f"<b>Overall Result:</b> {summary.get('overall_result', 'N/A').title()}",
                                        styles['Normal']))
            if summary.get("percentage_change") is not None:
                story.append(
                    self.Paragraph(f"<b>Performance Change:</b> {summary['percentage_change']:.2f}%", styles['Normal']))

            story.append(
                self.Paragraph(f"<b>Metrics Analyzed:</b> {summary.get('total_metrics', 0)}", styles['Normal']))
            story.append(
                self.Paragraph(f"<b>Improved Metrics:</b> {summary.get('improved_metrics', 0)}", styles['Normal']))
            story.append(
                self.Paragraph(f"<b>Regressed Metrics:</b> {summary.get('regressed_metrics', 0)}", styles['Normal']))

            if summary.get("matched_samples"):
                story.append(self.Paragraph(f"<b>Samples Compared:</b> {summary['matched_samples']}", styles['Normal']))

            story.append(self.Spacer(1, 20))

        # Compatibility Warnings
        if comparison_data.get("compatibility_warnings"):
            story.append(self.Paragraph("<b>Compatibility Warnings</b>", styles['Heading2']))
            for warning in comparison_data["compatibility_warnings"]:
                story.append(self.Paragraph(f"• {warning}", styles['Normal']))
            story.append(self.Spacer(1, 20))

        # Detailed Metric Analysis
        if comparison_data.get("comparison_results", {}).get("metric_comparison"):
            story.append(self.Paragraph("<b>Detailed Metric Analysis</b>", styles['Heading2']))

            # Create table data
            table_data = [["Metric", "Eval A Avg", "Eval B Avg", "Change (%)", "Effect Size", "Significant"]]

            metric_comparison = comparison_data["comparison_results"]["metric_comparison"]
            for metric_name, data in metric_comparison.items():
                if "comparison" in data and data["comparison"].get("absolute_difference") is not None:
                    avg_a = data["evaluation_a"].get("average", 0)
                    avg_b = data["evaluation_b"].get("average", 0)
                    pct_change = data["comparison"].get("percentage_change", 0)
                    effect_size = data["comparison"].get("effect_size")
                    is_significant = "Yes" if data["comparison"].get("is_significant") else "No"

                    # Handle None values in display
                    avg_a_str = f"{avg_a:.3f}" if avg_a is not None else "N/A"
                    avg_b_str = f"{avg_b:.3f}" if avg_b is not None else "N/A"
                    pct_change_str = f"{pct_change:.1f}%" if pct_change is not None else "N/A"
                    effect_size_str = f"{effect_size:.3f}" if effect_size is not None else "N/A"

                    table_data.append([
                        metric_name,
                        avg_a_str,
                        avg_b_str,
                        pct_change_str,
                        effect_size_str,
                        is_significant
                    ])

            if len(table_data) > 1:
                table = self.Table(table_data, hAlign='LEFT')
                table.setStyle(self.TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), self.colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), self.colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 9),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), self.colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, self.colors.black),
                    ('FONTSIZE', (0, 1), (-1, -1), 8),
                ]))
                story.append(table)
                story.append(self.Spacer(1, 20))

        # Natural Language Insights
        if comparison_data.get("narrative_insights"):
            story.append(self.Paragraph("<b>Insights and Recommendations</b>", styles['Heading2']))

            # Convert markdown-like formatting to reportlab formatting
            insights_text = comparison_data["narrative_insights"]
            # Basic conversion for bold text
            insights_text = insights_text.replace("**", "<b>").replace("**", "</b>")
            # Convert bullet points
            insights_lines = insights_text.split('\n')

            for line in insights_lines:
                if line.strip():
                    if line.startswith('- '):
                        story.append(self.Paragraph(f"• {line[2:]}", styles['Normal']))
                    elif line.startswith('#'):
                        # Convert headers
                        header_text = line.replace('#', '').strip()
                        story.append(self.Paragraph(header_text, styles['Heading3']))
                    else:
                        story.append(self.Paragraph(line, styles['Normal']))
                else:
                    story.append(self.Spacer(1, 6))

        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer


class ComparisonVisualizer:
    """Generate visualization data for comparison charts."""

    @staticmethod
    def generate_radar_chart_data(metric_comparison: Dict[str, Any], eval_a_name: str, eval_b_name: str) -> Dict[
        str, Any]:
        """Generate data for radar chart visualization."""
        labels = []
        series = [{"name": eval_a_name, "data": []}, {"name": eval_b_name, "data": []}]
        metric_is_inverted = []

        for metric_name, data in metric_comparison.items():
            if "evaluation_a" in data and "evaluation_b" in data:
                avg_a = data["evaluation_a"].get("average")
                avg_b = data["evaluation_b"].get("average")
                higher_is_better = data.get("config", {}).get("higher_is_better", True)

                if avg_a is not None and avg_b is not None:
                    labels.append(metric_name)

                    # For metrics where lower is better, invert for display
                    if not higher_is_better:
                        max_val = max(avg_a, avg_b, 1.0)
                        series[0]["data"].append(max_val - avg_a)
                        series[1]["data"].append(max_val - avg_b)
                        metric_is_inverted.append(True)
                    else:
                        series[0]["data"].append(avg_a)
                        series[1]["data"].append(avg_b)
                        metric_is_inverted.append(False)

        return {
            "type": "radar",
            "labels": labels,
            "series": series,
            "is_inverted": metric_is_inverted
        }

    @staticmethod
    def generate_bar_chart_data(metric_comparison: Dict[str, Any], eval_a_name: str, eval_b_name: str) -> Dict[
        str, Any]:
        """Generate data for bar chart visualization."""
        categories = []
        series = [
            {"name": eval_a_name, "data": []},
            {"name": eval_b_name, "data": []},
            {"name": "Change", "data": [], "type": "line"}
        ]
        significance = []
        higher_is_better = []

        for metric_name, data in metric_comparison.items():
            if "evaluation_a" in data and "evaluation_b" in data and "comparison" in data:
                avg_a = data["evaluation_a"].get("average")
                avg_b = data["evaluation_b"].get("average")

                if avg_a is not None and avg_b is not None:
                    categories.append(metric_name)
                    series[0]["data"].append(avg_a)
                    series[1]["data"].append(avg_b)

                    pct_change = data["comparison"].get("percentage_change", 0)
                    series[2]["data"].append(pct_change if pct_change is not None else 0)

                    significance.append(data["comparison"].get("is_significant", False))
                    higher_is_better.append(data.get("config", {}).get("higher_is_better", True))

        return {
            "type": "bar",
            "categories": categories,
            "series": series,
            "is_significant": significance,
            "higher_is_better": higher_is_better
        }