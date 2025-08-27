import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import joblib

class SmartBudgetManager:
    def __init__(self):
        self.festival_adjustments = {
            'Diwali': {'months': [10, 11], 'multiplier': 1.5, 'categories': ['Shopping', 'Food & Dining', 'Entertainment']},
            'Holi': {'months': [3], 'multiplier': 1.3, 'categories': ['Food & Dining', 'Entertainment']},
            'Eid': {'months': [4, 5, 6], 'multiplier': 1.4, 'categories': ['Shopping', 'Food & Dining']},
            'Christmas': {'months': [12], 'multiplier': 1.4, 'categories': ['Shopping', 'Entertainment']},
            'New Year': {'months': [1, 12], 'multiplier': 1.3, 'categories': ['Entertainment', 'Food & Dining']},
        }
        
        self.salary_cycle_patterns = {
            'beginning_of_month': {'days': range(1, 8), 'spending_multiplier': 1.2},
            'mid_month': {'days': range(8, 23), 'spending_multiplier': 1.0},
            'end_of_month': {'days': range(23, 32), 'spending_multiplier': 0.8},
        }
        
        self.base_budgets = {
            'Food & Dining': 8000,
            'Transportation': 5000,
            'Shopping': 6000,
            'Utilities': 4000,
            'Entertainment': 2000,
            'Healthcare': 3000,
            'EMI': 12000,
            'Others': 2000
        }
    
    def analyze_spending_patterns(self, transactions_df):
        """Analyze historical spending patterns"""
        if transactions_df.empty:
            return {}
        
        transactions_df['date'] = pd.to_datetime(transactions_df['date'])
        transactions_df['month'] = transactions_df['date'].dt.month
        transactions_df['day'] = transactions_df['date'].dt.day
        transactions_df['weekday'] = transactions_df['date'].dt.weekday
        
        patterns = {
            'monthly_trends': {},
            'weekly_patterns': {},
            'category_growth': {},
            'seasonal_variations': {}
        }
        
        monthly_spending = transactions_df[transactions_df['amount'] < 0].groupby(['month', 'category'])['amount'].sum().abs()
        for category in self.base_budgets.keys():
            if category in monthly_spending.index.get_level_values('category'):
                category_monthly = monthly_spending.xs(category, level='category')
                patterns['monthly_trends'][category] = category_monthly.to_dict()
        
        transactions_df['is_weekend'] = transactions_df['weekday'] >= 5
        weekend_spending = transactions_df[
            (transactions_df['amount'] < 0) & (transactions_df['is_weekend'])
        ]['amount'].sum()
        weekday_spending = transactions_df[
            (transactions_df['amount'] < 0) & (~transactions_df['is_weekend'])
        ]['amount'].sum()
        
        patterns['weekly_patterns'] = {
            'weekend_spending': abs(weekend_spending),
            'weekday_spending': abs(weekday_spending),
            'weekend_ratio': abs(weekend_spending) / (abs(weekday_spending) + abs(weekend_spending)) if (weekday_spending != 0) else 0
        }
        
        for category in self.base_budgets.keys():
            category_data = transactions_df[
                (transactions_df['category'] == category) & (transactions_df['amount'] < 0)
            ]
            if len(category_data) > 1:
                monthly_totals = category_data.groupby('month')['amount'].sum().abs()
                if len(monthly_totals) > 1:
                    growth_rate = (monthly_totals.iloc[-1] - monthly_totals.iloc[0]) / monthly_totals.iloc[0] * 100
                    patterns['category_growth'][category] = growth_rate
        
        return patterns
    
    def predict_next_month_spending(self, transactions_df, category=None):
        """Predict next month's spending using ML"""
        if transactions_df.empty:
            return self.base_budgets if category is None else self.base_budgets.get(category, 1000)
        
        transactions_df['date'] = pd.to_datetime(transactions_df['date'])
        
        monthly_data = transactions_df[transactions_df['amount'] < 0].groupby([
    transactions_df['date'].dt.year.rename('year'),
    transactions_df['date'].dt.month.rename('month'),
    'category'
])['amount'].sum().abs().reset_index()

        
        monthly_data.columns = ['year', 'month', 'category', 'amount']
        
        predictions = {}
        
        categories_to_predict = [category] if category else self.base_budgets.keys()
        
        for cat in categories_to_predict:
            cat_data = monthly_data[monthly_data['category'] == cat].copy()
            
            if len(cat_data) < 2:
                predictions[cat] = self.base_budgets.get(cat, 1000)
                continue
            
            cat_data['time_index'] = range(len(cat_data))
            cat_data['month_sin'] = np.sin(2 * np.pi * cat_data['month'] / 12)
            cat_data['month_cos'] = np.cos(2 * np.pi * cat_data['month'] / 12)
            
            if len(cat_data) >= 3:
                features = ['time_index', 'month_sin', 'month_cos']
                X = cat_data[features]
                y = cat_data['amount']
                
                model = RandomForestRegressor(n_estimators=50, random_state=42)
                model.fit(X, y)
                
                next_month = datetime.now().month
                next_time_index = len(cat_data)
                next_features = [[
                    next_time_index,
                    np.sin(2 * np.pi * next_month / 12),
                    np.cos(2 * np.pi * next_month / 12)
                ]]
                
                prediction = model.predict(next_features)[0]
                predictions[cat] = max(prediction, self.base_budgets.get(cat, 1000) * 0.5)
            
            else:
                X = cat_data[['time_index']]
                y = cat_data['amount']
                
                model = LinearRegression()
                model.fit(X, y)
                
                next_prediction = model.predict([[len(cat_data)]])[0]
                predictions[cat] = max(next_prediction, self.base_budgets.get(cat, 1000) * 0.5)
        
        return predictions[category] if category else predictions
    
    def generate_dynamic_budget(self, transactions_df, monthly_income=None, current_month=None):
        """Generate dynamic budget recommendations"""
        if current_month is None:
            current_month = datetime.now().month
        
        predicted_spending = self.predict_next_month_spending(transactions_df)
        
        festival_adjustments = {}
        for festival, config in self.festival_adjustments.items():
            if current_month in config['months']:
                for category in config['categories']:
                    if category in predicted_spending:
                        festival_adjustments[category] = config['multiplier']
        
        budget_recommendations = {}
        
        for category, predicted_amount in predicted_spending.items():
            base_budget = self.base_budgets.get(category, predicted_amount)
            
            festival_multiplier = festival_adjustments.get(category, 1.0)
            adjusted_prediction = predicted_amount * festival_multiplier
            
            recommended_budget = max(adjusted_prediction * 1.1, base_budget * 0.8)
            
            budget_recommendations[category] = {
                'predicted_spending': predicted_amount,
                'festival_adjustment': festival_multiplier,
                'recommended_budget': recommended_budget,
                'base_budget': base_budget,
                'adjustment_reason': self._get_adjustment_reason(category, current_month, festival_multiplier)
            }
        
        if monthly_income:
            total_recommended = sum(rec['recommended_budget'] for rec in budget_recommendations.values())
            if total_recommended > monthly_income * 0.8:  
                scale_factor = (monthly_income * 0.8) / total_recommended
                for category in budget_recommendations:
                    budget_recommendations[category]['recommended_budget'] *= scale_factor
                    budget_recommendations[category]['adjustment_reason'] += " (Income-adjusted)"
        
        return budget_recommendations
    
    def _get_adjustment_reason(self, category, month, festival_multiplier):
        """Get reason for budget adjustment"""
        reasons = []
        
        if festival_multiplier > 1.0:
            for festival, config in self.festival_adjustments.items():
                if month in config['months'] and category in config['categories']:
                    reasons.append(f"{festival} season")
        
        if month in [1, 12]:
            reasons.append("New Year period")
        elif month in [3, 4]:
            reasons.append("Financial year-end")
        elif month in [10, 11]:
            reasons.append("Festival season")
        
        return "; ".join(reasons) if reasons else "Based on spending patterns"
    
    def get_budget_alerts(self, transactions_df, current_budgets):
        """Generate budget alerts and recommendations"""
        current_month = datetime.now().month
        current_date = datetime.now()
        
        current_month_data = transactions_df[
            (pd.to_datetime(transactions_df['date']).dt.month == current_month) &
            (transactions_df['amount'] < 0)
        ]
        
        alerts = []
        
        if not current_month_data.empty:
            category_spending = current_month_data.groupby('category')['amount'].sum().abs()
            
            days_elapsed = current_date.day
            days_in_month = 30  
            month_progress = days_elapsed / days_in_month
            
            for category, budget in current_budgets.items():
                spent = category_spending.get(category, 0)
                spend_rate = spent / budget if budget > 0 else 0
                
                if spend_rate > 0.9:
                    alerts.append({
                        'type': 'danger',
                        'category': category,
                        'message': f"🚨 {category}: ₹{spent:,.0f}/₹{budget:,.0f} (90%+ used)",
                        'recommendation': f"Reduce {category.lower()} spending by ₹{(spent - budget * 0.8):,.0f}"
                    })
                elif spend_rate > 0.8:
                    alerts.append({
                        'type': 'warning',
                        'category': category,
                        'message': f"⚠️ {category}: ₹{spent:,.0f}/₹{budget:,.0f} (80%+ used)",
                        'recommendation': f"Monitor {category.lower()} spending closely"
                    })
                elif spend_rate > month_progress + 0.2: 
                    alerts.append({
                        'type': 'info',
                        'category': category,
                        'message': f"📊 {category}: Spending faster than expected",
                        'recommendation': f"Consider slowing down {category.lower()} expenses"
                    })
        
        return alerts

def smart_budget_page():
    """Smart budget management page"""
    st.header("🧠 Smart Budget Manager")
    st.markdown("**AI-powered dynamic budget recommendations**")
    
    if 'smart_budget' not in st.session_state:
        st.session_state.smart_budget = SmartBudgetManager()
    
    smart_budget = st.session_state.smart_budget
    
    data = load_transactions()
    
    st.subheader("💰 Monthly Income & Goals")
    col1, col2 = st.columns(2)
    
    with col1:
        monthly_income = st.number_input("Monthly Income (₹)", min_value=10000, value=50000, step=5000)
    
    with col2:
        savings_target = st.slider("Savings Target (%)", 10, 50, 20)
    
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
                        
                        if rec['festival_adjustment'] > 1.0:
                            st.warning(f"🎉 Festival adjustment: {rec['festival_adjustment']:.1f}x multiplier")
                    
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
                
                actual_savings_rate = (estimated_savings / monthly_income) * 100
                if actual_savings_rate >= savings_target:
                    st.success(f"🎉 Great! You're on track to save {actual_savings_rate:.1f}% of your income")
                else:
                    shortfall = savings_target - actual_savings_rate
                    st.warning(f"⚠️ You need to reduce spending by ₹{(shortfall * monthly_income / 100):,.0f} to meet your {savings_target}% savings target")
        
        st.subheader("🚨 Current Month Alerts")
        current_budgets = {cat: 5000 for cat in smart_budget.base_budgets.keys()} 
        alerts = smart_budget.get_budget_alerts(data, current_budgets)
        
        if alerts:
            for alert in alerts:
                if alert['type'] == 'danger':
                    st.error(f"{alert['message']}\n💡 {alert['recommendation']}")
                elif alert['type'] == 'warning':
                    st.warning(f"{alert['message']}\n💡 {alert['recommendation']}")
                else:
                    st.info(f"{alert['message']}\n💡 {alert['recommendation']}")
        else:
            st.success("✅ All categories within budget limits!")
        
        st.subheader("📊 Spending Pattern Analysis")
        patterns = smart_budget.analyze_spending_patterns(data)
        
        if patterns.get('weekly_patterns'):
            col1, col2 = st.columns(2)
            
            with col1:
                weekend_ratio = patterns['weekly_patterns']['weekend_ratio']
                st.metric("Weekend Spending Ratio", f"{weekend_ratio:.1%}")
                
                if weekend_ratio > 0.4:
                    st.info("💡 You spend significantly more on weekends. Consider setting weekend budgets.")
            
            with col2:
                if patterns['category_growth']:
                    fastest_growing = max(patterns['category_growth'], key=patterns['category_growth'].get)
                    growth_rate = patterns['category_growth'][fastest_growing]
                    st.metric("Fastest Growing Category", f"{fastest_growing} (+{growth_rate:.1f}%)")
    
    else:
        st.info("📊 Add more transactions to see smart budget recommendations!")
        
        st.subheader("🎯 Sample Smart Features")
        features = [
            "🎪 **Festival Adjustments**: Automatically increase budgets during Diwali, Eid, etc.",
            "📈 **Trend Analysis**: Predict spending based on historical patterns",
            "💰 **Income-based Scaling**: Ensure budgets don't exceed your income",
            "🚨 **Smart Alerts**: Get warned before overspending",
            "📊 **Pattern Recognition**: Identify weekend vs weekday spending habits",
            "🔮 **ML Predictions**: Use machine learning for accurate forecasts"
        ]
        
        for feature in features:
            st.markdown(f"• {feature}")
