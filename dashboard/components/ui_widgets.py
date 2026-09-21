"""
Reusable UI styling and custom components for Streamlit Dashboard.
"""

import streamlit as st


def apply_custom_styles():
    """Applies modern industrial theme styles."""
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
            
            html, body, [class*="css"] {
                font-family: 'Inter', sans-serif;
            }

            /* Custom Header Styling */
            .main-header {
                background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
                padding: 1.5rem 2rem;
                border-radius: 12px;
                border: 1px solid #334155;
                margin-bottom: 1.5rem;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
            }
            .main-header h1 {
                color: #F8FAFC;
                font-size: 1.8rem;
                font-weight: 700;
                margin: 0;
            }
            .main-header p {
                color: #94A3B8;
                font-size: 0.95rem;
                margin-top: 0.3rem;
                margin-bottom: 0;
            }

            /* Metric Card */
            .metric-card {
                background: #1E293B;
                border: 1px solid #334155;
                padding: 1.2rem;
                border-radius: 10px;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
                transition: transform 0.2s ease, border-color 0.2s ease;
            }
            .metric-card:hover {
                transform: translateY(-2px);
                border-color: #3B82F6;
            }
            .metric-title {
                color: #94A3B8;
                font-size: 0.82rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }
            .metric-value {
                color: #F8FAFC;
                font-size: 1.8rem;
                font-weight: 700;
                margin-top: 0.4rem;
            }
            .metric-sub {
                color: #64748B;
                font-size: 0.8rem;
                margin-top: 0.3rem;
            }

            /* Badge Tags */
            .badge-low {
                background-color: rgba(16, 185, 129, 0.15);
                color: #10B981;
                border: 1px solid #10B981;
                padding: 3px 8px;
                border-radius: 6px;
                font-size: 0.8rem;
                font-weight: 600;
            }
            .badge-medium {
                background-color: rgba(245, 158, 11, 0.15);
                color: #F59E0B;
                border: 1px solid #F59E0B;
                padding: 3px 8px;
                border-radius: 6px;
                font-size: 0.8rem;
                font-weight: 600;
            }
            .badge-high {
                background-color: rgba(239, 68, 68, 0.15);
                color: #EF4444;
                border: 1px solid #EF4444;
                padding: 3px 8px;
                border-radius: 6px;
                font-size: 0.8rem;
                font-weight: 600;
            }
        </style>
    """, unsafe_allow_html=True)


def render_header(title: str, subtitle: str, factory_id: str = "Factory_1"):
    """Renders sleek top banner."""
    st.markdown(f"""
        <div class="main-header">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h1>{title}</h1>
                    <p>{subtitle}</p>
                </div>
                <div style="background: #0284C7; color: white; padding: 6px 14px; border-radius: 20px; font-size: 0.85rem; font-weight: 600;">
                    Node: {factory_id}
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_metric_card(title: str, value: str, subtext: str = ""):
    """Renders responsive styled card."""
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{subtext}</div>
        </div>
    """, unsafe_allow_html=True)
