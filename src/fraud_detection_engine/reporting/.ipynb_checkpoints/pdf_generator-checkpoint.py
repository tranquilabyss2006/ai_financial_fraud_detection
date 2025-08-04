"""
PDF Generator Module
Implements PDF report generation for fraud detection results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import io
import base64
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.widgets import markers
import warnings
import logging
from typing import Dict, List, Tuple, Union

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFGenerator:
    """
    Class for generating PDF reports for fraud detection results
    Implements professional audit report generation
    """
    
    def __init__(self, output_dir='../reports/generated'):
        """
        Initialize PDFGenerator
        
        Args:
            output_dir (str): Output directory for reports
        """
        self.output_dir = output_dir
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def setup_custom_styles(self):
        """Setup custom styles for the report"""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        ))
        
        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='CustomSubtitle',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=12,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        ))
        
        # Section heading style
        self.styles.add(ParagraphStyle(
            name='SectionHeading',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            textColor=colors.darkblue
        ))
        
        # Subsection heading style
        self.styles.add(ParagraphStyle(
            name='SubsectionHeading',
            parent=self.styles['Heading3'],
            fontSize=12,
            spaceAfter=6,
            textColor=colors.darkblue
        ))
        
        # Body style
        self.styles.add(ParagraphStyle(
            name='Body',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=6,
            alignment=TA_LEFT
        ))
        
        # Footer style
        self.styles.add(ParagraphStyle(
            name='Footer',
            parent=self.styles['Normal'],
            fontSize=8,
            alignment=TA_CENTER,
            textColor=colors.grey
        ))
    
    def generate_executive_summary(self, df, risk_scores, top_fraud, include_charts=True, include_recommendations=True):
        """
        Generate executive summary report
        
        Args:
            df (DataFrame): Transaction data
            risk_scores (DataFrame): Risk scores
            top_fraud (DataFrame): Top fraudulent transactions
            include_charts (bool): Whether to include charts
            include_recommendations (bool): Whether to include recommendations
            
        Returns:
            str: Path to generated PDF
        """
        try:
            # Create filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"executive_summary_{timestamp}.pdf"
            filepath = os.path.join(self.output_dir, filename)
            
            # Create document
            doc = SimpleDocTemplate(filepath, pagesize=A4)
            story = []
            
            # Add title
            story.append(Paragraph("Fraud Detection Executive Summary", self.styles['CustomTitle']))
            story.append(Spacer(1, 12))
            
            # Add subtitle with date
            report_date = datetime.now().strftime("%B %d, %Y")
            story.append(Paragraph(f"Report Date: {report_date}", self.styles['CustomSubtitle']))
            story.append(PageBreak())
            
            # Add summary statistics
            story.append(Paragraph("Summary Statistics", self.styles['SectionHeading']))
            
            # Calculate statistics
            total_transactions = len(df)
            fraud_transactions = risk_scores['is_fraud'].sum()
            fraud_percentage = (fraud_transactions / total_transactions) * 100
            total_amount = df['amount'].sum() if 'amount' in df.columns else 0
            fraud_amount = df.loc[risk_scores['is_fraud'], 'amount'].sum() if 'amount' in df.columns else 0
            
            # Create statistics table
            stats_data = [
                ['Metric', 'Value'],
                ['Total Transactions', f"{total_transactions:,}"],
                ['Fraudulent Transactions', f"{fraud_transactions:,}"],
                ['Fraud Percentage', f"{fraud_percentage:.2f}%"],
                ['Total Amount', f"${total_amount:,.2f}"],
                ['Fraud Amount', f"${fraud_amount:,.2f}"]
            ]
            
            stats_table = Table(stats_data, colWidths=[3*inch, 2*inch])
            stats_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(stats_table)
            story.append(Spacer(1, 12))
            
            # Add risk distribution chart if requested
            if include_charts:
                story.append(Paragraph("Risk Distribution", self.styles['SectionHeading']))
                
                # Create risk distribution chart
                plt.figure(figsize=(8, 6))
                sns.histplot(risk_scores['risk_score'], bins=50, kde=True)
                plt.title('Distribution of Risk Scores')
                plt.xlabel('Risk Score')
                plt.ylabel('Frequency')
                
                # Convert plot to image
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
                img_buffer.seek(0)
                
                # Add image to report
                risk_dist_img = Image(img_buffer, width=6*inch, height=4*inch)
                story.append(risk_dist_img)
                story.append(Spacer(1, 12))
                
                plt.close()
            
            # Add top fraudulent transactions
            story.append(Paragraph("Top Fraudulent Transactions", self.styles['SectionHeading']))
            
            # Create table for top fraud transactions
            if len(top_fraud) > 0:
                # Prepare data
                fraud_data = []
                fraud_data.append(['Transaction ID', 'Amount', 'Risk Score', 'Date'])
                
                for _, row in top_fraud.head(10).iterrows():
                    transaction_id = row.get('transaction_id', 'N/A')
                    amount = f"${row.get('amount', 0):,.2f}"
                    risk_score = f"{row.get('risk_score', 0):.3f}"
                    date = row.get('timestamp', 'N/A')
                    
                    if isinstance(date, pd.Timestamp):
                        date = date.strftime('%Y-%m-%d')
                    
                    fraud_data.append([transaction_id, amount, risk_score, date])
                
                fraud_table = Table(fraud_data, colWidths=[1.5*inch, 1*inch, 1*inch, 1.5*inch])
                fraud_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                story.append(fraud_table)
                story.append(Spacer(1, 12))
            
            # Add recommendations if requested
            if include_recommendations:
                story.append(Paragraph("Recommendations", self.styles['SectionHeading']))
                
                recommendations = [
                    "1. Immediately review all high-risk transactions (risk score > 0.8).",
                    "2. Implement additional verification steps for transactions over $10,000.",
                    "3. Monitor patterns in fraudulent transactions to identify potential fraud rings.",
                    "4. Update fraud detection rules based on recent fraud patterns.",
                    "5. Conduct regular audits of the fraud detection system."
                ]
                
                for rec in recommendations:
                    story.append(Paragraph(rec, self.styles['Body']))
                
                story.append(Spacer(1, 12))
            
            # Add footer
            footer_text = f"Generated by Fraud Detection System on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            story.append(Paragraph(footer_text, self.styles['Footer']))
            
            # Build PDF
            doc.build(story)
            
            logger.info(f"Executive summary report generated: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error generating executive summary: {str(e)}")
            raise
    
    def generate_detailed_fraud_analysis(self, df, risk_scores, top_fraud, explanations, 
                                        include_charts=True, include_explanations=True, 
                                        include_recommendations=True):
        """
        Generate detailed fraud analysis report
        
        Args:
            df (DataFrame): Transaction data
            risk_scores (DataFrame): Risk scores
            top_fraud (DataFrame): Top fraudulent transactions
            explanations (dict): Transaction explanations
            include_charts (bool): Whether to include charts
            include_explanations (bool): Whether to include explanations
            include_recommendations (bool): Whether to include recommendations
            
        Returns:
            str: Path to generated PDF
        """
        try:
            # Create filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detailed_fraud_analysis_{timestamp}.pdf"
            filepath = os.path.join(self.output_dir, filename)
            
            # Create document
            doc = SimpleDocTemplate(filepath, pagesize=A4)
            story = []
            
            # Add title
            story.append(Paragraph("Detailed Fraud Analysis Report", self.styles['CustomTitle']))
            story.append(Spacer(1, 12))
            
            # Add subtitle with date
            report_date = datetime.now().strftime("%B %d, %Y")
            story.append(Paragraph(f"Report Date: {report_date}", self.styles['CustomSubtitle']))
            story.append(PageBreak())
            
            # Add table of contents
            story.append(Paragraph("Table of Contents", self.styles['SectionHeading']))
            
            toc_items = [
                "1. Executive Summary",
                "2. Methodology",
                "3. Analysis Results",
                "4. Detailed Transaction Analysis",
                "5. Risk Factors",
                "6. Recommendations",
                "7. Appendix"
            ]
            
            for item in toc_items:
                story.append(Paragraph(item, self.styles['Body']))
            
            story.append(PageBreak())
            
            # Add executive summary
            story.append(Paragraph("1. Executive Summary", self.styles['SectionHeading']))
            
            # Calculate statistics
            total_transactions = len(df)
            fraud_transactions = risk_scores['is_fraud'].sum()
            fraud_percentage = (fraud_transactions / total_transactions) * 100
            
            summary_text = f"""
            This report presents a detailed analysis of fraudulent transactions detected by the fraud detection system.
            The analysis identified {fraud_transactions:,} fraudulent transactions out of {total_transactions:,} total transactions,
            representing {fraud_percentage:.2f}% of all transactions. The total value of fraudulent transactions 
            amounts to ${df.loc[risk_scores['is_fraud'], 'amount'].sum():,.2f} if amount data is available.
            """
            
            story.append(Paragraph(summary_text, self.styles['Body']))
            story.append(Spacer(1, 12))
            
            # Add methodology
            story.append(Paragraph("2. Methodology", self.styles['SectionHeading']))
            
            methodology_text = """
            The fraud detection system employs a multi-layered approach combining unsupervised learning, 
            supervised learning, and rule-based methods to identify potentially fraudulent transactions. 
            The system analyzes various features including transaction amounts, frequency patterns, 
            geographic locations, temporal patterns, and behavioral characteristics to assign risk scores 
            to each transaction.
            """
            
            story.append(Paragraph(methodology_text, self.styles['Body']))
            story.append(Spacer(1, 12))
            
            # Add analysis results
            story.append(Paragraph("3. Analysis Results", self.styles['SectionHeading']))
            
            # Add risk distribution chart if requested
            if include_charts:
                story.append(Paragraph("3.1 Risk Distribution", self.styles['SubsectionHeading']))
                
                # Create risk distribution chart
                plt.figure(figsize=(8, 6))
                sns.histplot(risk_scores['risk_score'], bins=50, kde=True)
                plt.title('Distribution of Risk Scores')
                plt.xlabel('Risk Score')
                plt.ylabel('Frequency')
                
                # Convert plot to image
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
                img_buffer.seek(0)
                
                # Add image to report
                risk_dist_img = Image(img_buffer, width=6*inch, height=4*inch)
                story.append(risk_dist_img)
                story.append(Spacer(1, 12))
                
                plt.close()
            
            # Add fraud by time chart if requested
            if include_charts and 'timestamp' in df.columns:
                story.append(Paragraph("3.2 Fraud by Time", self.styles['SubsectionHeading']))
                
                # Create fraud by time chart
                df_with_risk = df.copy()
                df_with_risk['is_fraud'] = risk_scores['is_fraud']
                df_with_risk['timestamp'] = pd.to_datetime(df_with_risk['timestamp'])
                df_with_risk['date'] = df_with_risk['timestamp'].dt.date
                
                fraud_by_date = df_with_risk.groupby('date')['is_fraud'].agg(['sum', 'count']).reset_index()
                fraud_by_date['fraud_rate'] = fraud_by_date['sum'] / fraud_by_date['count']
                
                plt.figure(figsize=(10, 6))
                plt.subplot(2, 1, 1)
                plt.plot(fraud_by_date['date'], fraud_by_date['sum'], marker='o')
                plt.title('Fraud Count by Date')
                plt.ylabel('Count')
                plt.xticks(rotation=45)
                
                plt.subplot(2, 1, 2)
                plt.plot(fraud_by_date['date'], fraud_by_date['fraud_rate'], marker='o', color='red')
                plt.title('Fraud Rate by Date')
                plt.ylabel('Rate')
                plt.xlabel('Date')
                plt.xticks(rotation=45)
                
                plt.tight_layout()
                
                # Convert plot to image
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
                img_buffer.seek(0)
                
                # Add image to report
                fraud_time_img = Image(img_buffer, width=6*inch, height=5*inch)
                story.append(fraud_time_img)
                story.append(Spacer(1, 12))
                
                plt.close()
            
            story.append(PageBreak())
            
            # Add detailed transaction analysis
            story.append(Paragraph("4. Detailed Transaction Analysis", self.styles['SectionHeading']))
            
            # Add top fraudulent transactions
            story.append(Paragraph("4.1 Top Fraudulent Transactions", self.styles['SubsectionHeading']))
            
            # Create table for top fraud transactions
            if len(top_fraud) > 0:
                # Prepare data
                fraud_data = []
                fraud_data.append(['Transaction ID', 'Amount', 'Risk Score', 'Date', 'Risk Level'])
                
                for _, row in top_fraud.head(20).iterrows():
                    transaction_id = row.get('transaction_id', 'N/A')
                    amount = f"${row.get('amount', 0):,.2f}"
                    risk_score = f"{row.get('risk_score', 0):.3f}"
                    date = row.get('timestamp', 'N/A')
                    risk_level = row.get('risk_level', 'N/A')
                    
                    if isinstance(date, pd.Timestamp):
                        date = date.strftime('%Y-%m-%d')
                    
                    fraud_data.append([transaction_id, amount, risk_score, date, risk_level])
                
                fraud_table = Table(fraud_data, colWidths=[1.2*inch, 0.8*inch, 0.8*inch, 1*inch, 0.8*inch])
                fraud_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 8),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                story.append(fraud_table)
                story.append(Spacer(1, 12))
            
            # Add detailed explanations if requested
            if include_explanations and explanations:
                story.append(Paragraph("4.2 Transaction Explanations", self.styles['SubsectionHeading']))
                
                # Add explanations for top 5 fraudulent transactions
                for i, (idx, explanation) in enumerate(list(explanations.items())[:5]):
                    story.append(Paragraph(f"Transaction {i+1}: {explanation.get('transaction_id', idx)}", 
                                          self.styles['SubsectionHeading']))
                    
                    # Add explanation text
                    explanation_text = explanation.get('text_explanation', 'No explanation available.')
                    story.append(Paragraph(explanation_text, self.styles['Body']))
                    
                    # Add top factors
                    if explanation.get('top_factors'):
                        story.append(Paragraph("Top Contributing Factors:", self.styles['Body']))
                        for factor, contribution in explanation['top_factors'][:3]:
                            story.append(Paragraph(f"- {factor}: {contribution:.3f}", self.styles['Body']))
                    
                    # Add rule violations
                    if explanation.get('rule_violations'):
                        story.append(Paragraph("Rule Violations:", self.styles['Body']))
                        for rule in explanation['rule_violations']:
                            story.append(Paragraph(f"- {rule}", self.styles['Body']))
                    
                    story.append(Spacer(1, 12))
            
            story.append(PageBreak())
            
            # Add risk factors
            story.append(Paragraph("5. Risk Factors", self.styles['SectionHeading']))
            
            # Add common risk factors
            risk_factors = [
                "1. High Transaction Amount: Transactions significantly above average for the sender or receiver.",
                "2. Unusual Geographic Patterns: Transactions from or to high-risk locations.",
                "3. Temporal Anomalies: Transactions at unusual times or in rapid succession.",
                "4. Behavioral Deviations: Transactions that deviate from established patterns.",
                "5. Network Anomalies: Transactions involving suspicious networks of entities."
            ]
            
            for factor in risk_factors:
                story.append(Paragraph(factor, self.styles['Body']))
            
            story.append(Spacer(1, 12))
            
            # Add recommendations
            story.append(Paragraph("6. Recommendations", self.styles['SectionHeading']))
            
            recommendations = [
                "1. Immediate Actions:",
                "   - Block all high-risk transactions (risk score > 0.9).",
                "   - Contact customers associated with high-risk transactions for verification.",
                "   - Flag accounts involved in multiple suspicious transactions for review.",
                "",
                "2. System Improvements:",
                "   - Update fraud detection rules based on recent fraud patterns.",
                "   - Implement additional verification steps for high-value transactions.",
                "   - Enhance monitoring of cross-border transactions.",
                "",
                "3. Process Enhancements:",
                "   - Conduct regular audits of the fraud detection system.",
                "   - Provide training to staff on identifying fraud indicators.",
                "   - Establish clear procedures for handling suspected fraud.",
                "",
                "4. Long-term Strategies:",
                "   - Develop machine learning models to adapt to evolving fraud patterns.",
                "   - Implement real-time fraud detection capabilities.",
                "   - Collaborate with industry partners to share fraud intelligence."
            ]
            
            for rec in recommendations:
                story.append(Paragraph(rec, self.styles['Body']))
            
            story.append(Spacer(1, 12))
            
            # Add appendix
            story.append(Paragraph("7. Appendix", self.styles['SectionHeading']))
            
            # Add methodology details
            story.append(Paragraph("7.1 Methodology Details", self.styles['SubsectionHeading']))
            
            methodology_details = """
            The fraud detection system utilizes a combination of the following techniques:
            
            - Unsupervised Learning: Isolation Forest, Local Outlier Factor, Autoencoders
            - Supervised Learning: Random Forest, XGBoost, Neural Networks
            - Rule-based Detection: Configurable rules for known fraud patterns
            - Feature Engineering: Statistical, graph-based, NLP, and time-series features
            
            Each transaction is assigned a risk score between 0 and 1, with higher scores indicating 
            greater likelihood of fraud. Transactions with scores above the threshold (typically 0.5) 
            are flagged for review.
            """
            
            story.append(Paragraph(methodology_details, self.styles['Body']))
            story.append(Spacer(1, 12))
            
            # Add footer
            footer_text = f"Generated by Fraud Detection System on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            story.append(Paragraph(footer_text, self.styles['Footer']))
            
            # Build PDF
            doc.build(story)
            
            logger.info(f"Detailed fraud analysis report generated: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error generating detailed fraud analysis: {str(e)}")
            raise
    
    def generate_technical_report(self, df, risk_scores, model_results, include_charts=True):
        """
        Generate technical report
        
        Args:
            df (DataFrame): Transaction data
            risk_scores (DataFrame): Risk scores
            model_results (dict): Model results
            include_charts (bool): Whether to include charts
            
        Returns:
            str: Path to generated PDF
        """
        try:
            # Create filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"technical_report_{timestamp}.pdf"
            filepath = os.path.join(self.output_dir, filename)
            
            # Create document
            doc = SimpleDocTemplate(filepath, pagesize=A4)
            story = []
            
            # Add title
            story.append(Paragraph("Fraud Detection Technical Report", self.styles['CustomTitle']))
            story.append(Spacer(1, 12))
            
            # Add subtitle with date
            report_date = datetime.now().strftime("%B %d, %Y")
            story.append(Paragraph(f"Report Date: {report_date}", self.styles['CustomSubtitle']))
            story.append(PageBreak())
            
            # Add system overview
            story.append(Paragraph("1. System Overview", self.styles['SectionHeading']))
            
            overview_text = """
            The fraud detection system is designed to identify potentially fraudulent transactions 
            using a combination of machine learning techniques and rule-based methods. The system 
            processes transaction data in real-time, extracting various features and applying multiple 
            detection algorithms to generate risk scores for each transaction.
            """
            
            story.append(Paragraph(overview_text, self.styles['Body']))
            story.append(Spacer(1, 12))
            
            # Add data processing
            story.append(Paragraph("2. Data Processing", self.styles['SectionHeading']))
            
            data_processing_text = """
            The system processes transaction data through the following stages:
            
            1. Data Ingestion: Transactions are loaded from various sources (CSV, Excel, databases).
            2. Data Preprocessing: Missing values are handled, data types are converted, and basic 
               validation is performed.
            3. Feature Engineering: Statistical, graph-based, NLP, and time-series features are extracted.
            4. Model Application: Multiple models are applied to generate risk scores.
            5. Risk Aggregation: Scores from different models are combined to produce a final risk score.
            """
            
            story.append(Paragraph(data_processing_text, self.styles['Body']))
            story.append(Spacer(1, 12))
            
            # Add model performance
            story.append(Paragraph("3. Model Performance", self.styles['SectionHeading']))
            
            # Add unsupervised model performance
            if 'unsupervised' in model_results:
                story.append(Paragraph("3.1 Unsupervised Models", self.styles['SubsectionHeading']))
                
                unsupervised_text = f"""
                The system employs {len(model_results['unsupervised'])} unsupervised learning models:
                
                """
                
                for model_name in model_results['unsupervised']:
                    unsupervised_text += f"- {model_name}\n"
                
                story.append(Paragraph(unsupervised_text, self.styles['Body']))
                story.append(Spacer(1, 6))
            
            # Add supervised model performance
            if 'supervised' in model_results:
                story.append(Paragraph("3.2 Supervised Models", self.styles['SubsectionHeading']))
                
                supervised_text = f"""
                The system employs {len(model_results['supervised'])} supervised learning models:
                
                """
                
                for model_name in model_results['supervised']:
                    if 'performance' in model_results['supervised'][model_name]:
                        perf = model_results['supervised'][model_name]['performance']
                        auc = perf.get('roc_auc', 0)
                        supervised_text += f"- {model_name}: AUC = {auc:.3f}\n"
                    else:
                        supervised_text += f"- {model_name}\n"
                
                story.append(Paragraph(supervised_text, self.styles['Body']))
                story.append(Spacer(1, 6))
            
            # Add rule-based model performance
            if 'rule' in model_results:
                story.append(Paragraph("3.3 Rule-based Models", self.styles['SubsectionHeading']))
                
                rule_text = f"""
                The system employs {len(model_results['rule'].get('rules', {}))} rule-based detectors:
                
                """
                
                for rule_name in model_results['rule'].get('rules', {}):
                    rule_text += f"- {rule_name}\n"
                
                story.append(Paragraph(rule_text, self.styles['Body']))
                story.append(Spacer(1, 6))
            
            # Add feature importance if available
            if 'supervised' in model_results:
                story.append(Paragraph("3.4 Feature Importance", self.styles['SubsectionHeading']))
                
                # Get feature importance from Random Forest if available
                if 'random_forest' in model_results['supervised']:
                    rf_data = model_results['supervised']['random_forest']
                    if 'feature_importance' in rf_data:
                        importance_df = rf_data['feature_importance']
                        
                        # Create table for top features
                        feature_data = []
                        feature_data.append(['Feature', 'Importance'])
                        
                        for _, row in importance_df.head(10).iterrows():
                            feature_data.append([row['feature'], f"{row['importance']:.3f}"])
                        
                        feature_table = Table(feature_data, colWidths=[3*inch, 1.5*inch])
                        feature_table.setStyle(TableStyle([
                            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                            ('FONTSIZE', (0, 0), (-1, 0), 10),
                            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                            ('GRID', (0, 0), (-1, -1), 1, colors.black)
                        ]))
                        
                        story.append(feature_table)
                        story.append(Spacer(1, 6))
            
            # Add system architecture
            story.append(Paragraph("4. System Architecture", self.styles['SectionHeading']))
            
            architecture_text = """
            The fraud detection system is built with a modular architecture consisting of the following components:
            
            1. Data Ingestion Layer: Handles data loading from various sources.
            2. Feature Engineering Layer: Extracts features for fraud detection.
            3. Model Layer: Applies various detection algorithms.
            4. Risk Aggregation Layer: Combines model outputs.
            5. Reporting Layer: Generates reports and visualizations.
            6. API Layer: Provides interfaces for external systems.
            """
            
            story.append(Paragraph(architecture_text, self.styles['Body']))
            story.append(Spacer(1, 12))
            
            # Add performance metrics
            story.append(Paragraph("5. Performance Metrics", self.styles['SectionHeading']))
            
            # Calculate performance metrics
            total_transactions = len(df)
            fraud_transactions = risk_scores['is_fraud'].sum()
            fraud_percentage = (fraud_transactions / total_transactions) * 100
            
            metrics_data = [
                ['Metric', 'Value'],
                ['Total Transactions', f"{total_transactions:,}"],
                ['Fraudulent Transactions', f"{fraud_transactions:,}"],
                ['Fraud Percentage', f"{fraud_percentage:.2f}%"],
                ['Processing Time', f"{len(df) * 0.001:.2f} seconds (estimated)"]
            ]
            
            metrics_table = Table(metrics_data, colWidths=[3*inch, 2*inch])
            metrics_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(metrics_table)
            story.append(Spacer(1, 12))
            
            # Add footer
            footer_text = f"Generated by Fraud Detection System on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            story.append(Paragraph(footer_text, self.styles['Footer']))
            
            # Build PDF
            doc.build(story)
            
            logger.info(f"Technical report generated: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error generating technical report: {str(e)}")
            raise
    
    def generate_custom_report(self, df, risk_scores, top_fraud, explanations, model_results, 
                             include_charts=True, include_explanations=True, 
                             include_recommendations=True):
        """
        Generate custom report with user-specified sections
        
        Args:
            df (DataFrame): Transaction data
            risk_scores (DataFrame): Risk scores
            top_fraud (DataFrame): Top fraudulent transactions
            explanations (dict): Transaction explanations
            model_results (dict): Model results
            include_charts (bool): Whether to include charts
            include_explanations (bool): Whether to include explanations
            include_recommendations (bool): Whether to include recommendations
            
        Returns:
            str: Path to generated PDF
        """
        try:
            # Create filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"custom_report_{timestamp}.pdf"
            filepath = os.path.join(self.output_dir, filename)
            
            # Create document
            doc = SimpleDocTemplate(filepath, pagesize=A4)
            story = []
            
            # Add title
            story.append(Paragraph("Custom Fraud Detection Report", self.styles['CustomTitle']))
            story.append(Spacer(1, 12))
            
            # Add subtitle with date
            report_date = datetime.now().strftime("%B %d, %Y")
            story.append(Paragraph(f"Report Date: {report_date}", self.styles['CustomSubtitle']))
            story.append(PageBreak())
            
            # Add summary statistics
            story.append(Paragraph("Summary Statistics", self.styles['SectionHeading']))
            
            # Calculate statistics
            total_transactions = len(df)
            fraud_transactions = risk_scores['is_fraud'].sum()
            fraud_percentage = (fraud_transactions / total_transactions) * 100
            
            stats_text = f"""
            Total Transactions: {total_transactions:,}
            Fraudulent Transactions: {fraud_transactions:,}
            Fraud Percentage: {fraud_percentage:.2f}%
            """
            
            story.append(Paragraph(stats_text, self.styles['Body']))
            story.append(Spacer(1, 12))
            
            # Add risk distribution chart if requested
            if include_charts:
                story.append(Paragraph("Risk Distribution", self.styles['SectionHeading']))
                
                # Create risk distribution chart
                plt.figure(figsize=(8, 6))
                sns.histplot(risk_scores['risk_score'], bins=50, kde=True)
                plt.title('Distribution of Risk Scores')
                plt.xlabel('Risk Score')
                plt.ylabel('Frequency')
                
                # Convert plot to image
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
                img_buffer.seek(0)
                
                # Add image to report
                risk_dist_img = Image(img_buffer, width=6*inch, height=4*inch)
                story.append(risk_dist_img)
                story.append(Spacer(1, 12))
                
                plt.close()
            
            # Add top fraudulent transactions
            story.append(Paragraph("Top Fraudulent Transactions", self.styles['SectionHeading']))
            
            # Create table for top fraud transactions
            if len(top_fraud) > 0:
                # Prepare data
                fraud_data = []
                fraud_data.append(['Transaction ID', 'Amount', 'Risk Score'])
                
                for _, row in top_fraud.head(10).iterrows():
                    transaction_id = row.get('transaction_id', 'N/A')
                    amount = f"${row.get('amount', 0):,.2f}"
                    risk_score = f"{row.get('risk_score', 0):.3f}"
                    
                    fraud_data.append([transaction_id, amount, risk_score])
                
                fraud_table = Table(fraud_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
                fraud_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                story.append(fraud_table)
                story.append(Spacer(1, 12))
            
            # Add detailed explanations if requested
            if include_explanations and explanations:
                story.append(Paragraph("Transaction Explanations", self.styles['SectionHeading']))
                
                # Add explanations for top 3 fraudulent transactions
                for i, (idx, explanation) in enumerate(list(explanations.items())[:3]):
                    story.append(Paragraph(f"Transaction {i+1}", self.styles['SubsectionHeading']))
                    
                    # Add explanation text
                    explanation_text = explanation.get('text_explanation', 'No explanation available.')
                    story.append(Paragraph(explanation_text, self.styles['Body']))
                    story.append(Spacer(1, 6))
            
            # Add recommendations if requested
            if include_recommendations:
                story.append(Paragraph("Recommendations", self.styles['SectionHeading']))
                
                recommendations = [
                    "1. Review all high-risk transactions immediately.",
                    "2. Implement additional verification for large transactions.",
                    "3. Monitor patterns in fraudulent activity.",
                    "4. Update detection rules regularly.",
                    "5. Conduct periodic system audits."
                ]
                
                for rec in recommendations:
                    story.append(Paragraph(rec, self.styles['Body']))
                
                story.append(Spacer(1, 12))
            
            # Add footer
            footer_text = f"Generated by Fraud Detection System on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            story.append(Paragraph(footer_text, self.styles['Footer']))
            
            # Build PDF
            doc.build(story)
            
            logger.info(f"Custom report generated: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error generating custom report: {str(e)}")
            raise