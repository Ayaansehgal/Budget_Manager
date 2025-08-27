import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
import time
import json
import numpy as np
import re

try:
    from voice_handler import VoiceExpenseHandler, VOICE_EXAMPLES
    from ocr_handler import ReceiptOCRHandler
    from upi_integration import AdvancedUPIHandler
    from smart_budget import SmartBudgetManager
    PHASE1_AVAILABLE = True
except ImportError:
    PHASE1_AVAILABLE = False
    st.warning("Phase 1 modules not found. Some features will be disabled.")

st.set_page_config(
    page_title="🤖 AI Budget Manager",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #FF9933, #FFFFFF, #138808);
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    margin-bottom: 20px;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 20px;
    border-radius: 10px;
    border-left: 5px solid #FF9933;
}
.stButton > button {
    background-color: #FF9933;
    color: white;
    border-radius: 20px;
    border: none;
    padding: 0.5rem 1rem;
}
.stButton > button:hover {
    background-color: #e8850e;
    color: white;
}
</style>
""", unsafe_allow_html=True)

API_BASE = "http://localhost:8000"

INDIAN_CATEGORIES = {
    "Food & Dining": ["ZOMATO", "SWIGGY", "UBER EATS", "DOMINOS", "MCDONALD", "KFC", "BIG BAZAAR", "DMART"],
    "Transportation": ["UBER", "OLA", "METRO", "IRCTC", "INDIAN OIL", "BHARAT PETROLEUM", "HP PETROL"],
    "Utilities": ["BSEB", "MSEB", "AIRTEL", "JIO", "VODAFONE", "TATA SKY", "DISH TV", "ACT FIBERNET"],
    "Shopping": ["AMAZON", "FLIPKART", "MYNTRA", "AJIO", "APOLLO PHARMACY", "MEDPLUS"],
    "Entertainment": ["NETFLIX", "AMAZON PRIME", "HOTSTAR", "PVR", "INOX", "BOOKMYSHOW"],
    "Healthcare": ["APOLLO", "FORTIS", "MAX", "MEDPLUS", "NETMEDS", "PRACTO"],
    "Education": ["BYJU", "UNACADEMY", "COURSERA", "UDEMY"],
    "Investments": ["SBI MF", "ICICI PRUDENTIAL", "HDFC AMC", "ZERODHA", "GROWW", "PAYTM MONEY"],
    "EMI": ["EMI", "LOAN", "BAJAJ FINSERV", "HDFC BANK", "SBI", "ICICI"],
    "Others": []
}

@st.cache_data
def load_transactions():
    try:
        df = pd.read_csv("data/transactions.csv")
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        return df
    except FileNotFoundError:
        return pd.DataFrame(columns=[
            'date', 'description', 'amount', 'category', 'type', 'confidence'
        ])

def save_transaction(transaction_data):
    try:
        existing_data = pd.read_csv("data/transactions.csv")
        new_data = pd.concat([existing_data, pd.DataFrame([transaction_data])], ignore_index=True)
    except FileNotFoundError:
        new_data = pd.DataFrame([transaction_data])
    
    import os
    os.makedirs("data", exist_ok=True)
    
    new_data.to_csv("data/transactions.csv", index=False)

def call_api(endpoint, method="GET", data=None):
    try:
        if method == "POST":
            response = requests.post(f"{API_BASE}{endpoint}", json=data)
        else:
            response = requests.get(f"{API_BASE}{endpoint}")
        
        return response.json() if response.status_code == 200 else None
    except:
        return None

def categorize_transaction(description: str) -> str:
    """Basic categorization function"""
    desc_upper = description.upper()
    for category, keywords in INDIAN_CATEGORIES.items():
        for keyword in keywords:
            if keyword in desc_upper:
                return category
    return "Others"

def voice_expense_page():
    """Voice expense entry page"""
    st.header("🎤 Voice Expense Entry")
    st.markdown("**Speak naturally in Hindi or English to add expenses!**")
    
    if not PHASE1_AVAILABLE:
        st.error("❌ Voice recognition module not available. Please install dependencies.")
        st.code("pip install SpeechRecognition pyttsx3 pyaudio")
        return
    
    if 'voice_handler' not in st.session_state:
        st.session_state.voice_handler = VoiceExpenseHandler()
    
    voice_handler = st.session_state.voice_handler
    
    with st.expander("📝 Voice Command Examples"):
        st.markdown("**Try saying:**")
        for example in VOICE_EXAMPLES:
            st.markdown(f"• *\"{example}\"*")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🎤 Start Voice Recording", type="primary", use_container_width=True):
            with st.spinner("🎧 Listening..."):
                text = voice_handler.listen_for_expense()
                
                if text == "timeout":
                    st.warning("⏰ No speech detected. Please try again.")
                elif text == "unclear":
                    st.error("🔇 Could not understand speech. Please speak clearly.")
                elif text == "error":
                    st.error("❌ Speech recognition service error.")
                else:
                    st.success(f"🎯 Heard: \"{text}\"")
                    
                    expense_data = voice_handler.parse_expense_from_text(text)
                    
                    st.subheader("📊 Extracted Details")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if expense_data['amount']:
                            st.metric("💰 Amount", f"₹{expense_data['amount']:,}")
                        else:
                            st.warning("💰 Amount not detected")
                    
                    with col2:
                        st.metric("📁 Category", expense_data['category'])
                    
                    with col3:
                        confidence_color = "🟢" if expense_data['confidence'] > 0.7 else "🟡"
                        st.metric("🎯 Confidence", f"{confidence_color} {expense_data['confidence']:.1f}")
                    
                    st.subheader("✏️ Review & Confirm")
                    
                    with st.form("voice_expense_form"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            final_description = st.text_input("Description", value=expense_data['description'])
                            final_amount = st.number_input("Amount (₹)", value=expense_data['amount'] or 0, step=1)
                        
                        with col2:
                            categories_list = list(INDIAN_CATEGORIES.keys())
                            try:
                                default_index = categories_list.index(expense_data['category'])
                            except ValueError:
                                default_index = len(categories_list) - 1  # "Others"
                            
                            final_category = st.selectbox("Category", categories_list, index=default_index)
                            final_date = st.date_input("Date", datetime.now())
                        
                        if st.form_submit_button("✅ Add Voice Expense", type="primary"):
                            if final_amount > 0:
                                transaction_data = {
                                    'date': final_date.strftime('%Y-%m-%d'),
                                    'description': final_description,
                                    'amount': -final_amount,  
                                    'category': final_category,
                                    'type': 'Voice Entry',
                                    'confidence': expense_data['confidence']
                                }
                                
                                save_transaction(transaction_data)
                                
                                voice_handler.speak_confirmation({
                                    'amount': final_amount,
                                    'category': final_category
                                })
                                
                                st.success(f"🎉 Voice expense added: ₹{final_amount} for {final_category}")
                                time.sleep(2)
                                st.rerun()
                            else:
                                st.error("Please enter a valid amount")
    
    with col2:
        st.info("💡 **Tips for better recognition:**\n\n• Speak clearly and slowly\n• Mention amount in rupees\n• Include merchant name\n• Use simple words")

def receipt_scanning_page():
    """Receipt scanning page"""
    st.header("📷 Receipt Scanner")
    st.markdown("**Scan receipts to automatically extract expense details**")
    
    if not PHASE1_AVAILABLE:
        st.error("❌ OCR module not available. Please install dependencies.")
        st.code("pip install opencv-python pytesseract Pillow")
        return
    
    if 'ocr_handler' not in st.session_state:
        st.session_state.ocr_handler = ReceiptOCRHandler()
    
    ocr_handler = st.session_state.ocr_handler
    
    uploaded_file = st.file_uploader(
        "Choose a receipt image...", 
        type=['png', 'jpg', 'jpeg'],
        help="Upload a clear photo of your receipt"
    )
    
    if uploaded_file is not None:
        from PIL import Image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption="Uploaded Receipt", use_container_width=True)
        
        with col2:
            if st.button("🔍 Scan Receipt", type="primary"):
                with st.spinner("🔍 Scanning receipt..."):
                    ocr_text, confidence = ocr_handler.extract_text_from_image(image)
                    
                    if ocr_text.strip():
                        receipt_data = ocr_handler.parse_receipt_data(ocr_text)
                        category = ocr_handler.categorize_from_receipt(receipt_data)
                        
                        st.success("✅ Receipt scanned successfully!")
                        
                        st.subheader("📊 Extracted Information")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            if receipt_data['amount']:
                                st.metric("💰 Amount", f"₹{receipt_data['amount']:,.2f}")
                            else:
                                st.warning("Amount not detected")
                        
                        with col2:
                            if receipt_data['merchant']:
                                st.metric("🏪 Merchant", receipt_data['merchant'][:20])
                            else:
                                st.warning("Merchant not detected")
                        
                        with col3:
                            st.metric("📁 Category", category)
                        
                        st.subheader("✏️ Review & Add Expense")
                        
                        with st.form("receipt_expense_form"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                final_amount = st.number_input(
                                    "Amount (₹)", 
                                    value=receipt_data['amount'] or 0, 
                                    step=0.01
                                )
                                final_description = st.text_input(
                                    "Description", 
                                    value=receipt_data['merchant'] or "Receipt expense"
                                )
                            
                            with col2:
                                categories_list = list(INDIAN_CATEGORIES.keys())
                                try:
                                    default_index = categories_list.index(category)
                                except ValueError:
                                    default_index = len(categories_list) - 1
                                
                                final_category = st.selectbox("Category", categories_list, index=default_index)
                                final_date = st.date_input("Date", datetime.now())
                            
                            if st.form_submit_button("✅ Add Receipt Expense", type="primary"):
                                if final_amount > 0:
                                    transaction_data = {
                                        'date': final_date.strftime('%Y-%m-%d'),
                                        'description': final_description,
                                           'amount': -final_amount,
                                        'category': final_category,
                                        'type': 'Receipt Scan',
                                        'confidence': receipt_data['confidence']
                                    }
                                    
                                    save_transaction(transaction_data)
                                    st.success(f"🎉 Receipt expense added: ₹{final_amount} for {final_category}")
                                    time.sleep(2)
                                    st.rerun()
                                else:
                                    st.error("Please enter a valid amount")
                    else:
                        st.error("❌ Could not extract text from receipt. Try a clearer image.")

def upi_integration_page():
    """UPI integration page"""
    st.header("💳 Advanced UPI Integration")
    st.markdown("**Monitor and analyze your UPI spending patterns**")
    
    if not PHASE1_AVAILABLE:
        st.error("❌ UPI module not available.")
        return
    
    if 'upi_handler' not in st.session_state:
        st.session_state.upi_handler = AdvancedUPIHandler()
    
    upi_handler = st.session_state.upi_handler
    
    st.subheader("Parse UPI Notification")
    
    sample_notifications = [
        "You paid ₹250 to ZOMATO using Google Pay on 27-Aug-25. UPI Ref No 425678943210",
        "₹150 sent to OLA CABS via PhonePe. Transaction successful. UTR: 425678943211",
        "₹500 paid to AMAZON PAY INDIA using Paytm. Txn ID: 425678943212",
        "₹75 transferred to UBER INDIA via BHIM UPI. Ref: 425678943213"
    ]
    
    notification_text = st.text_area(
        "Paste UPI notification here:",
        height=100,
        placeholder="Paste your UPI payment notification text here..."
    )
    
    with st.expander("📋 Try Sample Notifications"):
        for i, sample in enumerate(sample_notifications, 1):
            if st.button(f"Use Sample {i}", key=f"sample_{i}"):
                st.session_state.notification_text = sample
                st.rerun()
    
    if notification_text.strip():
        if st.button("🔍 Parse Notification", type="primary"):
            parsed_transaction = upi_handler.parse_upi_notification(notification_text)
            
            if parsed_transaction['amount']:
                st.success(" UPI transaction parsed successfully!")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(" Amount", f"₹{parsed_transaction['amount']:,.2f}")
                
                with col2:
                    st.metric(" App", parsed_transaction['app'] or 'Unknown')
                
                with col3:
                    merchant_display = parsed_transaction['merchant'] or 'Unknown'
                    if len(merchant_display) > 15:
                        merchant_display = merchant_display[:15] + "..."
                    st.metric(" Merchant", merchant_display)
                
                with col4:
                    confidence_color = "🟢" if parsed_transaction['confidence'] > 0.8 else "🟡"
                    st.metric(" Confidence", f"{confidence_color} {parsed_transaction['confidence']:.1f}")
                
                if st.button("➕ Add to Expenses", type="secondary"):
                    transaction_data = {
                        'date': datetime.now().strftime('%Y-%m-%d'),
                        'description': f"{parsed_transaction['app']} - {parsed_transaction['merchant']}",
                        'amount': -parsed_transaction['amount'],
                        'category': parsed_transaction['category'],
                        'type': f"UPI - {parsed_transaction['app']}",
                        'confidence': parsed_transaction['confidence']
                    }
                    
                    save_transaction(transaction_data)
                    st.success("🎉 UPI transaction added to expenses!")
            
            else:
                st.error("❌ Could not parse UPI transaction. Please check the notification format.")

def smart_budget_page():
    """Smart budget management page"""
    st.header("🧠 Smart Budget Manager")
    st.markdown("**AI-powered dynamic budget recommendations**")
    
    if not PHASE1_AVAILABLE:
        st.error("❌ Smart budget module not available.")
        return
    
    if 'smart_budget' not in st.session_state:
        st.session_state.smart_budget = SmartBudgetManager()
    
    smart_budget = st.session_state.smart_budget
    
    st.subheader("💰 Monthly Income & Goals")
    col1, col2 = st.columns(2)
    
    with col1:
        monthly_income = st.number_input("Monthly Income (₹)", min_value=10000, value=50000, step=5000)
    
    with col2:
        savings_target = st.slider("Savings Target (%)", 10, 50, 20)
    
    data = load_transactions()
    
    if not data.empty:
        if st.button("🔮 Generate Smart Budget", type="primary"):
            with st.spinner("🧠 Analyzing spending patterns..."):
                budget_recommendations = smart_budget.generate_dynamic_budget(data, monthly_income)
                
                st.success("✅ Smart budget generated!")
                
                st.subheader("📊 AI Budget Recommendations")
                
                total_recommended = 0
                for category, rec in budget_recommendations.items():
                    with st.expander(f"💳 {category} - ₹{rec['recommended_budget']:,.0f}"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Predicted Spending", f"₹{rec['predicted_spending']:,.0f}")
                        
                        with col2:
                            st.metric("Recommended Budget", f"₹{rec['recommended_budget']:,.0f}")
                        
                        with col3:
                            change = ((rec['recommended_budget'] - rec['base_budget']) / rec['base_budget']) * 100
                            st.metric("vs Base Budget", f"{change:+.1f}%")
                        
                        st.info(f"**Reason:** {rec['adjustment_reason']}")
                    
                    total_recommended += rec['recommended_budget']
                
                st.subheader("📈 Budget Summary")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Recommended Budget", f"₹{total_recommended:,.0f}")
                
                with col2:
                    budget_percentage = (total_recommended / monthly_income) * 100
                    st.metric("% of Income", f"{budget_percentage:.1f}%")
                
                with col3:
                    estimated_savings = monthly_income - total_recommended
                    st.metric("Estimated Savings", f"₹{estimated_savings:,.0f}")
    else:
        st.info("📊 Add more transactions to see smart budget recommendations!")

def main():
    st.markdown('<div class="main-header"><h1>🇮🇳 भारतीय AI Budget Tracker</h1><p>Track करें, Save करें, Invest करें!</p></div>', unsafe_allow_html=True)
    
    st.sidebar.title("🧭 Navigation")
    st.sidebar.markdown("---")
    
    if PHASE1_AVAILABLE:
       page_options = [
    "🏠 Dashboard", "➕ Add Transaction", "🧠 Smart Budget", "🧠 AI Insights", "📊 ML Predictions", "💰 SIP Planner", "⚙️ Settings"
    ]

    else:
        page_options = [
            "🏠 Dashboard", "➕ Add Transaction", "🧠 AI Insights", 
            "📊 ML Predictions", "💰 SIP Planner", "⚙️ Settings"
        ]
    
    page = st.sidebar.selectbox("Choose Section:", page_options)
    
    if page == "🏠 Dashboard":
        st.header("📊 Smart Financial Dashboard")
        
        data = load_transactions()
        
        if not data.empty:
            ai_insights = call_api("/smart_insights")
            
            col1, col2, col3, col4 = st.columns(4)
            
            current_month = datetime.now().month
            current_year = datetime.now().year
            current_month_data = data[
                (data['date'].dt.month == current_month) & 
                (data['date'].dt.year == current_year)
            ]
            
            income = current_month_data[current_month_data['amount'] > 0]['amount'].sum()
            expenses = abs(current_month_data[current_month_data['amount'] < 0]['amount'].sum())
            savings = income - expenses
            savings_rate = (savings / income * 100) if income > 0 else 0
            
            with col1:
                st.metric("💵 Monthly Income", f"₹{income:,.0f}")
            
            with col2:
                st.metric("💸 Monthly Expenses", f"₹{expenses:,.0f}")
            
            with col3:
                st.metric("💰 Monthly Savings", f"₹{savings:,.0f}", f"{savings_rate:.1f}%")
            
            with col4:
                st.metric("📝 Transactions", len(current_month_data))
            
            st.markdown("---")
            
            if ai_insights and ai_insights.get("alerts"):
                st.subheader("🚨 AI Alerts")
                for alert in ai_insights["alerts"]:
                    st.warning(alert)
            
            col1, col2 = st.columns(2)
            
            with col1:
                expense_data = current_month_data[current_month_data['amount'] < 0].copy()
                if not expense_data.empty:
                    expense_data['amount'] = expense_data['amount'].abs()
                    category_totals = expense_data.groupby('category')['amount'].sum().reset_index()
                    
                    fig_pie = px.pie(
                        category_totals,
                        values='amount',
                        names='category',
                        title="🍕 Monthly Expenses by Category"
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                if not expense_data.empty:
                    daily_expenses = expense_data.groupby(expense_data['date'].dt.date)['amount'].sum().reset_index()
                    daily_expenses.columns = ['date', 'amount']
                    
                    fig_line = px.line(
                        daily_expenses,
                        x='date',
                        y='amount',
                        title="📈 Daily Spending Trend",
                        labels={'date': 'Date', 'amount': 'Amount (₹)'}
                    )
                    fig_line.update_traces(line_color='#FF6B6B')
                    st.plotly_chart(fig_line, use_container_width=True)
            
            if ai_insights and ai_insights.get("recommendations"):
                st.subheader("💡 AI Recommendations")
                for rec in ai_insights["recommendations"]:
                    st.markdown(f"• {rec}")
            
            st.subheader("🕒 Recent Transactions")
            recent = data.tail(10).sort_values('date', ascending=False)
            st.dataframe(recent[['date', 'description', 'amount', 'category', 'type']], use_container_width=True)
        
        else:
            st.info("📝 No data yet. Add transactions to see AI insights!")
            
            st.subheader("🤖 Available AI Features")
            features = [
                "🎯 **Spending Predictions**: ML forecasts your monthly expenses",
                "🚨 **Anomaly Detection**: Identifies unusual transactions", 
                "💰 **Budget Optimization**: AI suggests optimal budget allocation",
                "📊 **Behavioral Analysis**: Insights into your spending patterns",
                "🏆 **Smart Recommendations**: Personalized money-saving tips"
            ]
            
            if PHASE1_AVAILABLE:
                features.extend([
                    "🎤 **Voice Commands**: Add expenses using voice",
                    "📷 **Receipt Scanning**: OCR-based expense extraction",
                    "💳 **UPI Integration**: Parse payment app notifications",
                    "🧠 **Smart Budgets**: Dynamic AI budget recommendations"
                ])
            
            for feature in features:
                st.markdown(feature)
    
    elif page == "➕ Add Transaction":
        st.header("➕ Add New Transaction with AI Categorization")
        
        tab1, tab2 = st.tabs(["Manual Entry", "Quick Add"])
        
        with tab1:
            with st.form("transaction_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    description = st.text_input("📝 Description", placeholder="e.g., Zomato food delivery")
                    amount = st.number_input("💰 Amount (₹)", step=1.0)
                    transaction_type = st.selectbox("Type", ["Expense (-)", "Income (+)"])
                
                with col2:
                    date = st.date_input("📅 Date", datetime.now())
                    use_ai = st.checkbox("🤖 Use AI Categorization", value=True)
                
                submitted = st.form_submit_button("➕ Add Transaction", type="primary")
                
                if submitted and description and amount:
                    final_amount = -abs(amount) if transaction_type == "Expense (-)" else abs(amount)
                    
                    if use_ai:
                        prediction = call_api("/predict_category", "POST", {
                            "description": description,
                            "amount": abs(amount)
                        })
                        
                        if prediction:
                            category = prediction['category']
                            confidence = prediction['confidence']
                            st.success(f"🤖 AI Prediction: {category} (confidence: {confidence:.2f})")
                        else:
                            category = categorize_transaction(description)
                            confidence = 0.8
                    else:
                        category = categorize_transaction(description)
                        confidence = 1.0
                    
                    transaction_data = {
                        'date': date.strftime('%Y-%m-%d'),
                        'description': description,
                        'amount': final_amount,
                        'category': category,
                        'type': transaction_type,
                        'confidence': confidence
                    }
                    
                    save_transaction(transaction_data)
                    st.success(f"✅ Transaction added! Category: {category}")
                    time.sleep(1)
                    st.rerun()
        
        with tab2:
            st.markdown("### 🚀 Quick Expense Entry")
            quick_categories = ["Food & Dining", "Transportation", "Shopping", "Entertainment"]
            
            cols = st.columns(4)
            for i, cat in enumerate(quick_categories):
                with cols[i]:
                    if st.button(f"{cat}\n₹500", key=f"quick_{cat}"):
                        transaction_data = {
                            'date': datetime.now().strftime('%Y-%m-%d'),
                            'description': f"Quick {cat} expense",
                            'amount': -500,
                            'category': cat,
                            'type': "Expense (-)",
                            'confidence': 1.0
                        }
                        save_transaction(transaction_data)
                        st.success(f"✅ Added ₹500 {cat} expense!")
                        time.sleep(1)
                        st.rerun()

    
    elif page == "🧠 Smart Budget" and PHASE1_AVAILABLE:
        smart_budget_page()
    elif page == "🧠 AI Insights":
        st.header("🧠 AI-Powered Financial Insights")
        
        insights = call_api("/smart_insights")
        
        if insights:
            st.subheader("📊 Spending Analysis")
            summary = insights.get("spending_summary", {})
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Expenses", f"₹{summary.get('total_expenses', 0):,.0f}")
            with col2:
                st.metric("Avg Transaction", f"₹{summary.get('average_transaction', 0):,.0f}")
            with col3:
                st.metric("Top Category", summary.get('top_category', 'None'))
            
            if insights.get("recommendations"):
                st.subheader("🎯 Personalized Recommendations")
                for i, rec in enumerate(insights["recommendations"], 1):
                    st.markdown(f"**{i}.** {rec}")
            
            anomalies = call_api("/anomaly_detection")
            if anomalies and anomalies.get("anomalies"):
                st.subheader("🔍 Unusual Spending Detected")
                st.warning(f"Found {anomalies['anomalies_found']} unusual transactions")
                
                for anomaly in anomalies["anomalies"]:
                    with st.expander(f"₹{abs(anomaly['transaction']['amount']):,.0f} - {anomaly['transaction']['description']}"):
                        st.write(f"**Reason:** {anomaly['reason']}")
                        st.write(f"**Date:** {anomaly['transaction']['date']}")
                        st.write(f"**Anomaly Score:** {anomaly['anomaly_score']:.2f}")
        else:
            st.error("Could not fetch AI insights. Make sure the API is running.")
    
    elif page == "📊 ML Predictions":
        st.header("📊 Machine Learning Predictions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Spending Forecast")
            
            categories_list = ["All Categories"] + list(INDIAN_CATEGORIES.keys())
            category = st.selectbox("Select Category", categories_list)
            
            months_ahead = st.slider("Months to Predict", 1, 6, 1)
            
            if st.button("🔮 Generate Prediction"):
                prediction = call_api("/predict_spending", "POST", {
                    "category": None if category == "All Categories" else category,
                    "months_ahead": months_ahead
                })
                
                if prediction:
                    st.success("✅ Prediction Generated!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Predicted Amount", f"₹{prediction['predicted_amount']:,.0f}")
                    with col2:
                        st.metric("Budget Breach Risk", f"{prediction['budget_breach_probability']:.1f}%")
                    with col3:
                        confidence_color = "🟢" if prediction['confidence'] == 'high' else "🟡"
                        st.metric("Confidence", f"{confidence_color} {prediction['confidence']}")
                    
                    risk = prediction['budget_breach_probability']
                    if risk > 80:
                        st.error("🚨 High risk of budget breach! Consider reducing expenses.")
                    elif risk > 60:
                        st.warning("⚠️ Moderate risk. Monitor spending closely.")
                    else:
                        st.success("✅ Spending looks healthy!")
                    
                    st.markdown(f"**AI Recommendation:** {prediction['recommendation']}")
        
        with col2:
            st.subheader("📈 Trend Analysis")
            
            data = load_transactions()
            if not data.empty:
                data_temp = data.copy()
                data_temp['year_month'] = data_temp['date'].dt.to_period('M').astype(str)
                monthly_data = data_temp.groupby('year_month')['amount'].sum().reset_index()
                
                fig = px.line(
                    monthly_data,
                    x='year_month',
                    y='amount',
                    title="Monthly Spending Trend"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    elif page == "💰 SIP Planner":
        st.header("💰 SIP Investment Planner")
        st.markdown("Plan your Systematic Investment Plans for financial goals")
        
        tab1, tab2 = st.tabs(["Goal Planner", "SIP Calculator"])
        
        with tab1:
            st.subheader("🎯 Set Financial Goals")
            
            with st.form("sip_goal_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    goal_name = st.selectbox("Goal", [
                        "Emergency Fund", "Home Down Payment", "Car Purchase", 
                        "Child Education", "Retirement", "Vacation", "Wedding"
                    ])
                    target_amount = st.number_input("Target Amount (₹)", min_value=50000, value=1000000, step=50000)
                
                with col2:
                    tenure_years = st.slider("Investment Tenure (Years)", 1, 30, 10)
                    risk_profile = st.selectbox("Risk Profile", ["conservative", "moderate", "aggressive"])
                
                calculate_goal = st.form_submit_button("📊 Calculate SIP", type="primary")
                
                if calculate_goal:
                    result = call_api("/calculate_sip", "POST", {
                        "goal_name": goal_name,
                        "target_amount": target_amount,
                        "tenure_years": tenure_years,
                        "risk_profile": risk_profile
                    })
                    
                    if result:
                        st.success("✅ SIP Calculation Complete!")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("💰 Required Monthly SIP", f"₹{result['required_monthly_sip']:,.0f}")
                        
                        with col2:
                            st.metric("💸 Total Investment", f"₹{result['total_investment']:,.0f}")
                        
                        with col3:
                            st.metric("🎯 Maturity Value", f"₹{result['maturity_value']:,.0f}")
                        
                        with col4:
                            returns = result['maturity_value'] - result['total_investment']
                            return_pct = (returns / result['total_investment']) * 100
                            st.metric("📈 Returns", f"₹{returns:,.0f}", f"{return_pct:.1f}%")
        
        with tab2:
            st.subheader("🧮 SIP Calculator")
            
            col1, col2 = st.columns(2)
            
            with col1:
                monthly_sip = st.number_input("Monthly SIP Amount (₹)", min_value=500, value=5000, step=500)
                calc_tenure = st.slider("Tenure (Years)", 1, 30, 15)
                expected_return = st.slider("Expected Annual Return (%)", 6, 20, 12)
            
            with col2:
                months = calc_tenure * 12
                monthly_return = expected_return / 100 / 12
                
                future_value = monthly_sip * (((1 + monthly_return) ** months - 1) / monthly_return) * (1 + monthly_return)
                total_invested = monthly_sip * months
                returns = future_value - total_invested
                
                st.metric("Total Investment", f"₹{total_invested:,.0f}")
                st.metric("Maturity Value", f"₹{future_value:,.0f}")
                st.metric("Total Returns", f"₹{returns:,.0f}")
                st.metric("Return Multiple", f"{future_value/total_invested:.1f}x")
    
    elif page == "⚙️ Settings":
        st.header("⚙️ Settings & Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🤖 System Status")
            
            api_status = call_api("/")
            if api_status:
                st.success(f"✅ API Connected - Version {api_status.get('version', 'Unknown')}")
                
                ml_status = "🟢 Trained" if api_status.get('ml_status') else "🟡 Learning"
                st.info(f"ML Model Status: {ml_status}")
                
                data = load_transactions()
                st.info(f"Training Data: {len(data)} transactions")
            else:
                st.error("❌ API Not Connected")
            
            if PHASE1_AVAILABLE:
                st.success("✅ Phase 1 Features Available")
                st.markdown("• Voice Commands\n• Receipt Scanning\n• UPI Integration\n• Smart Budgets")
            else:
                st.warning("⚠️ Phase 1 Features Not Available")
        
        with col2:
            st.subheader("💰 Budget Settings")
            
            categories = list(INDIAN_CATEGORIES.keys())
            
            for category in categories:
                budget_limit = st.number_input(
                    f"{category} Monthly Budget (₹)",
                    min_value=0,
                    value=5000,
                    step=500,
                    key=f"budget_{category}"
                )
    
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            "<div style='text-align: center'>"
            f"<p>🤖 AI-Powered Budget Management | Made by Ayaan Sehgal: {'✅ Active' if PHASE1_AVAILABLE else '⚠️ Partial'}</p>"
            "</div>",
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()