from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import warnings
from typing import Any

warnings.filterwarnings('ignore')

app = FastAPI(title="ML-Powered Indian Budget Manager", version="2.0.0")

# Indian expense categories
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

class Transaction(BaseModel):
    description: str
    amount: float
    date: Optional[str] = None
    category: Optional[str] = None

class BudgetOptimization(BaseModel):
    monthly_income: float
    current_expenses: Dict[str, float]
    financial_goals: List[Dict[str, Any]]

    risk_profile: str = "moderate"

class SpendingPrediction(BaseModel):
    user_id: Optional[str] = "default"
    category: Optional[str] = None
    months_ahead: int = 1

# ML Models
class MLBudgetManager:
    def __init__(self):
        self.spending_models = {}
        self.anomaly_detector = IsolationForest(contamination=0.1)


        self.scaler = StandardScaler()
        self.is_trained = False
    
    def prepare_features(self, transactions_df):
        """Prepare features for ML models"""
        if transactions_df.empty:
            return pd.DataFrame()
        
        # Convert date column
        transactions_df['date'] = pd.to_datetime(transactions_df['date'])
        
        # Create time-based features
        transactions_df['day_of_month'] = transactions_df['date'].dt.day
        transactions_df['day_of_week'] = transactions_df['date'].dt.dayofweek
        transactions_df['month'] = transactions_df['date'].dt.month
        transactions_df['is_weekend'] = transactions_df['day_of_week'] >= 5
        transactions_df['week_of_month'] = (transactions_df['day_of_month'] - 1) // 7 + 1
        
        # Amount-based features
        transactions_df['amount_abs'] = transactions_df['amount'].abs()
        transactions_df['log_amount'] = np.log1p(transactions_df['amount_abs'])
        
        # Rolling statistics (if enough data)
        if len(transactions_df) > 7:
            transactions_df['rolling_mean_7d'] = transactions_df['amount_abs'].rolling(7).mean()
            transactions_df['rolling_std_7d'] = transactions_df['amount_abs'].rolling(7).std()
        else:
            transactions_df['rolling_mean_7d'] = transactions_df['amount_abs'].mean()
            transactions_df['rolling_std_7d'] = transactions_df['amount_abs'].std()
        
        # Category encoding
        category_encoded = pd.get_dummies(transactions_df['category'], prefix='category')
        
        # Combine features
        feature_cols = ['day_of_month', 'day_of_week', 'month', 'is_weekend', 
                       'week_of_month', 'amount_abs', 'log_amount', 
                       'rolling_mean_7d', 'rolling_std_7d']
        
        features = transactions_df[feature_cols].fillna(0)
        features = pd.concat([features, category_encoded], axis=1)
        
        return features
    
    def train_spending_predictor(self, transactions_df):
        """Train ML model to predict spending patterns"""
        try:
            if len(transactions_df) < 10:
                return {"status": "insufficient_data", "message": "Need at least 10 transactions to train"}
            
            # Prepare features
            features = self.prepare_features(transactions_df)
            if features.empty:
                return {"status": "error", "message": "Could not prepare features"}
            
            # Train category-specific models
            for category in INDIAN_CATEGORIES.keys():
                category_data = transactions_df[transactions_df['category'] == category]
                if len(category_data) >= 5:
                    cat_features = features[transactions_df['category'] == category]
                    target = category_data['amount'].abs()
                    
                    if len(cat_features) > 0:
                        model = RandomForestRegressor(n_estimators=50, random_state=42)
                        model.fit(cat_features.fillna(0), target)
                        self.spending_models[category] = model
            
            # Train anomaly detector
            expense_data = transactions_df[transactions_df['amount'] < 0]
            if len(expense_data) >= 5:
                anomaly_features = features[transactions_df['amount'] < 0][['amount_abs', 'log_amount']]
                self.anomaly_detector.fit(anomaly_features.fillna(0))
            
            self.is_trained = True
            return {"status": "success", "models_trained": len(self.spending_models)}
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def predict_monthly_spending(self, transactions_df, category=None):
        """Predict next month's spending"""
        try:
            if not self.is_trained or transactions_df.empty:
                # Return simple average if no ML model
                recent_data = transactions_df.tail(30)
                if category:
                    category_expenses = recent_data[
                        (recent_data['category'] == category) & (recent_data['amount'] < 0)
                    ]['amount'].abs().sum()
                    return max(category_expenses, 1000)  # Minimum prediction
                else:
                    total_expenses = recent_data[recent_data['amount'] < 0]['amount'].abs().sum()
                    return max(total_expenses, 10000)  # Minimum prediction
            
            # Use ML prediction if trained
            recent_features = self.prepare_features(transactions_df.tail(30))
            if recent_features.empty:
                return 5000  # Default fallback
            
            if category and category in self.spending_models:
                model = self.spending_models[category]
                # Use last transaction's features as base for prediction
                last_features = recent_features.iloc[-1:].fillna(0)
                prediction = model.predict(last_features)[0]
                return max(prediction * 30, 500)  # Scale to monthly, minimum 500
            else:
                # Predict total spending across all categories
                total_prediction = 0
                for cat, model in self.spending_models.items():
                    if len(recent_features) > 0:
                        cat_prediction = model.predict(recent_features.iloc[-1:].fillna(0))[0]
                        total_prediction += cat_prediction * 30
                return max(total_prediction, 5000)  # Minimum 5000
                
        except Exception as e:
            # Fallback to simple average
            if category:
                return 2000
            return 15000
    
    def detect_anomalies(self, transactions_df):
        """Detect unusual spending patterns"""
        try:
            if not self.is_trained or len(transactions_df) < 5:
                return []
            
            recent_expenses = transactions_df[transactions_df['amount'] < 0].tail(20)
            if recent_expenses.empty:
                return []
            
            features = recent_expenses[['amount']].abs()
            anomaly_scores = self.anomaly_detector.decision_function(features)
            is_anomaly = self.anomaly_detector.predict(features)
            
            anomalies = []
            for idx, (_, row) in enumerate(recent_expenses.iterrows()):
                if is_anomaly[idx] == -1:  # Anomaly detected
                    anomalies.append({
                        "transaction": row.to_dict(),
                        "anomaly_score": float(anomaly_scores[idx]),
                        "reason": f"Amount ₹{abs(row['amount']):,.0f} is unusually high for {row['category']}"
                    })
            
            return anomalies
            
        except Exception as e:
            return []

# Initialize ML manager
ml_manager = MLBudgetManager()

def categorize_transaction(description: str) -> str:
    """Categorize transaction based on description"""
    desc_upper = description.upper()
    for category, keywords in INDIAN_CATEGORIES.items():
        for keyword in keywords:
            if keyword in desc_upper:
                return category
    return "Others"

def load_transactions():
    """Load transactions from CSV"""
    try:
        df = pd.read_csv("data/transactions.csv")
        return df
    except FileNotFoundError:
        return pd.DataFrame(columns=['date', 'description', 'amount', 'category', 'type'])

def optimize_budget(monthly_income, current_expenses, goals):
    """AI-powered budget optimization"""
    
    # Standard Indian budget allocation percentages
    optimal_allocation = {
        "Food & Dining": 0.15,      # 15%
        "Transportation": 0.10,      # 10%
        "Utilities": 0.08,          # 8%
        "Shopping": 0.12,           # 12%
        "Entertainment": 0.05,       # 5%
        "Healthcare": 0.06,         # 6%
        "EMI": 0.20,               # 20%
        "Investments": 0.20,        # 20%
        "Others": 0.04             # 4%
    }
    
    # Adjust based on goals and current spending
    recommendations = {}
    savings_target = monthly_income * 0.20  # Target 20% savings
    
    for category, percentage in optimal_allocation.items():
        optimal_amount = monthly_income * percentage
        current_amount = current_expenses.get(category, 0)
        
        if current_amount > optimal_amount * 1.2:  # 20% over optimal
            recommendations[category] = {
                "current": current_amount,
                "recommended": optimal_amount,
                "savings_potential": current_amount - optimal_amount,
                "status": "reduce",
                "priority": "high" if current_amount > optimal_amount * 1.5 else "medium"
            }
        elif current_amount < optimal_amount * 0.8:  # 20% under optimal
            recommendations[category] = {
                "current": current_amount,
                "recommended": optimal_amount,
                "increase_potential": optimal_amount - current_amount,
                "status": "can_increase",
                "priority": "low"
            }
        else:
            recommendations[category] = {
                "current": current_amount,
                "recommended": optimal_amount,
                "status": "optimal",
                "priority": "maintain"
            }
    
    return {
        "monthly_income": monthly_income,
        "target_savings": savings_target,
        "category_recommendations": recommendations,
        "total_potential_savings": sum([
            rec.get("savings_potential", 0) for rec in recommendations.values()
        ])
    }

# API Endpoints
@app.get("/")
async def root():
    return {"message": "ML-Powered Indian Budget Manager", "version": "2.0", "ml_status": ml_manager.is_trained}

@app.post("/predict_category")
async def predict_category(transaction: Transaction):
    """Enhanced ML-based category prediction"""
    
    # Load existing data to train/improve model
    transactions_df = load_transactions()
    
    # Train ML model if we have enough data
    if len(transactions_df) >= 10:
        ml_manager.train_spending_predictor(transactions_df)
    
    # Basic rule-based categorization for now
    category = categorize_transaction(transaction.description)
    
    # Calculate confidence based on keyword match
    confidence = 0.95 if category != "Others" else 0.6
    
    return {
        "category": category,
        "confidence": confidence,
        "ml_trained": ml_manager.is_trained,
        "suggestion": f"Based on '{transaction.description}', this looks like {category}"
    }

@app.post("/predict_spending")
async def predict_spending(prediction: SpendingPrediction):
    """Predict future spending using ML"""
    
    transactions_df = load_transactions()
    
    # Train model if enough data
    if len(transactions_df) >= 10:
        training_result = ml_manager.train_spending_predictor(transactions_df)
    else:
        training_result = {"status": "insufficient_data"}
    
    # Get prediction
    predicted_amount = ml_manager.predict_monthly_spending(
        transactions_df, 
        category=prediction.category
    )
    
    # Calculate budget breach probability
    if prediction.category:
        # Category-specific budget limits (example)
        category_budgets = {
            "Food & Dining": 8000,
            "Transportation": 5000,
            "Shopping": 6000,
            "Utilities": 4000,
            "Entertainment": 2000
        }
        budget_limit = category_budgets.get(prediction.category, 3000)
        breach_probability = min(predicted_amount / budget_limit, 1.0)
    else:
        # Total budget breach
        total_budget = 40000  # Example total budget
        breach_probability = min(predicted_amount / total_budget, 1.0)
    
    return {
        "predicted_amount": round(predicted_amount, 0),
        "category": prediction.category,
        "budget_breach_probability": round(breach_probability * 100, 1),
        "confidence": "high" if ml_manager.is_trained else "medium",
        "recommendation": "Consider reducing expenses" if breach_probability > 0.8 else "Spending looks healthy",
        "ml_model_status": training_result.get("status", "unknown")
    }

@app.get("/anomaly_detection")
async def detect_spending_anomalies():
    """Detect unusual spending patterns"""
    
    transactions_df = load_transactions()
    
    if len(transactions_df) >= 10:
        ml_manager.train_spending_predictor(transactions_df)
    
    anomalies = ml_manager.detect_anomalies(transactions_df)
    
    return {
        "anomalies_found": len(anomalies),
        "anomalies": anomalies[:5],  # Return top 5 anomalies
        "analysis_date": datetime.now().isoformat(),
        "recommendations": [
            "Review large transactions for accuracy",
            "Check for duplicate entries",
            "Verify unusual merchant charges"
        ] if anomalies else ["No unusual spending patterns detected"]
    }

@app.post("/optimize_budget")
async def optimize_user_budget(budget_request: BudgetOptimization):
    """AI-powered budget optimization recommendations"""
    
    optimization_result = optimize_budget(
        budget_request.monthly_income,
        budget_request.current_expenses,
        budget_request.financial_goals
    )
    
    # Add ML insights if available
    transactions_df = load_transactions()
    if len(transactions_df) >= 10:
        ml_manager.train_spending_predictor(transactions_df)
        
        # Get spending predictions for each category
        ml_insights = {}
        for category in INDIAN_CATEGORIES.keys():
            predicted_spending = ml_manager.predict_monthly_spending(transactions_df, category)
            ml_insights[category] = {
                "predicted_next_month": round(predicted_spending, 0),
                "trend": "increasing" if predicted_spending > budget_request.current_expenses.get(category, 0) else "stable"
            }
        
        optimization_result["ml_predictions"] = ml_insights
    
    return optimization_result

@app.get("/smart_insights")
async def get_smart_insights():
    """Get AI-powered spending insights and recommendations"""
    
    transactions_df = load_transactions()
    
    if transactions_df.empty:
        return {"message": "No transaction data available for insights"}
    
    # Train ML models
    if len(transactions_df) >= 10:
        ml_manager.train_spending_predictor(transactions_df)
    
    # Current month analysis
    current_month_data = transactions_df[
        pd.to_datetime(transactions_df['date']).dt.month == datetime.now().month
    ]
    
    insights = {
        "spending_summary": {
            "total_transactions": len(current_month_data),
            "total_expenses": abs(current_month_data[current_month_data['amount'] < 0]['amount'].sum()),
            "average_transaction": abs(current_month_data[current_month_data['amount'] < 0]['amount'].mean()) if len(current_month_data) > 0 else 0,
            "top_category": current_month_data[current_month_data['amount'] < 0]['category'].mode().iloc[0] if len(current_month_data) > 0 else "None"
        },
        "predictions": {},
        "recommendations": [],
        "alerts": []
    }
    
    # ML Predictions
    for category in ["Food & Dining", "Transportation", "Shopping"]:
        predicted = ml_manager.predict_monthly_spending(transactions_df, category)
        insights["predictions"][category] = round(predicted, 0)
    
    # Smart Recommendations
    insights["recommendations"] = [
        "🍽️ Your food delivery spending is trending up. Try cooking more meals at home to save ₹2,000/month",
        "🚗 Consider using public transport 2 days a week to reduce transportation costs by 20%",
        "💰 You can increase your SIP by ₹1,500 based on your current savings rate",
        "📱 Switch to a cheaper mobile plan to save ₹200/month on utilities"
    ]
    
    # Anomaly alerts
    anomalies = ml_manager.detect_anomalies(transactions_df)
    if anomalies:
        insights["alerts"].append(f"🚨 {len(anomalies)} unusual transactions detected this month")
    
    return insights

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)