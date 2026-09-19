import streamlit_app as st
import pandas as pd
import requests
from datetime import datetime


st.set_page_config(
    page_title="Finlora Fraud Detection",
    page_icon="💳",
    layout="wide"
)


API_URL = "http://127.0.0.1:8000/predict"

CSV_PATH = r"C:\Users\HP\Downloads\Fraudulent_Transaction_Detection_For_Finlora_Company\Finlora_Dataset\FinLora_Customer_Transaction_Dataset.csv"


@st.cache_data
def load_data():
    df = pd.read_csv(CSV_PATH)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce"
        )

    return df


try:
    historical_data = load_data()

except Exception as e:
    st.error(f"Unable to load historical dataset: {e}")
    st.stop()


def get_unique_values(column_name):

    if column_name not in historical_data.columns:
        return []

    values = (
        historical_data[column_name]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    return sorted(values)


def get_customer_record(customer_id):

    customer_records = historical_data[
        historical_data["customer_id"].astype(str) == str(customer_id)
    ].copy()

    if customer_records.empty:
        return None

    if "timestamp" in customer_records.columns:
        customer_records = customer_records.sort_values(
            "timestamp",
            ascending=False
        )

    return customer_records.iloc[0]


st.title("💳 Finlora Fraud Detection Dashboard")

st.markdown(
    """
    Use this dashboard to submit a transaction for fraud-risk analysis.
    Customer information and categorical values are loaded dynamically
    from the historical Finlora transaction dataset.
    """
)


col1, col2, col3 = st.columns(3)


with col1:
    st.metric(
        "Historical Transactions",
        f"{len(historical_data):,}"
    )


with col2:

    if "customer_id" in historical_data.columns:
        number_customers = historical_data["customer_id"].nunique()
    else:
        number_customers = 0

    st.metric(
        "Customers",
        f"{number_customers:,}"
    )


with col3:

    if "is_fraud" in historical_data.columns:
        fraud_count = historical_data["is_fraud"].sum()
    else:
        fraud_count = 0

    st.metric(
        "Historical Fraud Cases",
        f"{fraud_count:,}"
    )


st.divider()


st.subheader("Customer Information")


if "customer_id" not in historical_data.columns:
    st.error("The historical dataset does not contain customer_id.")
    st.stop()


customer_ids = (
    historical_data["customer_id"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)


customer_ids = sorted(customer_ids)


selected_customer = st.selectbox(
    "Customer ID",
    options=customer_ids
)


customer_record = get_customer_record(
    selected_customer
)


if customer_record is None:
    st.warning(
        "No historical information was found for this customer."
    )
    st.stop()


st.subheader("Preloaded Customer Information")


customer_col1, customer_col2, customer_col3 = st.columns(3)


with customer_col1:

    home_country_values = get_unique_values(
        "home_country"
    )

    customer_home_country = str(
        customer_record["home_country"]
    )

    if customer_home_country in home_country_values:

        home_country = st.selectbox(
            "Home Country",
            home_country_values,
            index=home_country_values.index(
                customer_home_country
            )
        )

    else:

        home_country = st.selectbox(
            "Home Country",
            home_country_values
        )


with customer_col2:

    kyc_values = get_unique_values(
        "kyc_tier"
    )

    customer_kyc = str(
        customer_record["kyc_tier"]
    )

    if customer_kyc in kyc_values:

        kyc_tier = st.selectbox(
            "KYC Tier",
            kyc_values,
            index=kyc_values.index(
                customer_kyc
            )
        )

    else:

        kyc_tier = st.selectbox(
            "KYC Tier",
            kyc_values
        )


with customer_col3:

    account_age = int(
        customer_record["account_age_days"]
    )

    st.number_input(
        "Account Age (days)",
        min_value=0,
        value=account_age,
        disabled=True
    )


risk_col1, risk_col2, risk_col3 = st.columns(3)


with risk_col1:

    device_trust = float(
        customer_record["device_trust_score"]
    )

    st.number_input(
        "Device Trust Score",
        value=device_trust,
        disabled=True
    )


with risk_col2:

    internal_risk = float(
        customer_record["risk_score_internal"]
    )

    st.number_input(
        "Internal Risk Score",
        value=internal_risk,
        disabled=True
    )


with risk_col3:

    ip_country_values = get_unique_values(
        "ip_country"
    )

    customer_ip_country = str(
        customer_record["ip_country"]
    )

    if customer_ip_country in ip_country_values:

        ip_country_index = ip_country_values.index(
            customer_ip_country
        )

    else:

        ip_country_index = 0

    ip_country = st.selectbox(
        "IP Country",
        ip_country_values,
        index=ip_country_index
    )


st.divider()


st.subheader("Transaction Information")


transaction_col1, transaction_col2 = st.columns(2)


with transaction_col1:

    timestamp = st.date_input(
        "Transaction Date",
        value=datetime.now().date()
    )


    source_currency_values = get_unique_values(
        "source_currency"
    )


    source_currency = st.selectbox(
        "Source Currency",
        source_currency_values
    )


    amount_sc = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=100.0,
        step=10.0
    )


    fee = st.number_input(
        "Transaction Fee",
        min_value=0.0,
        value=0.0,
        step=0.01
    )


with transaction_col2:

    dest_currency_values = get_unique_values(
        "dest_currency"
    )


    dest_currency = st.selectbox(
        "Destination Currency",
        dest_currency_values
    )


    channel_values = get_unique_values(
        "channel"
    )


    channel = st.selectbox(
        "Payment Channel",
        channel_values
    )


    corridor_risk_default = float(
        customer_record["corridor_risk"]
        if "corridor_risk" in customer_record.index
        else 0.0
    )


    corridor_risk = st.number_input(
        "Corridor Risk",
        min_value=0.0,
        value=corridor_risk_default,
        step=0.01
    )


st.subheader("Transaction Risk Information")


risk_col1, risk_col2, risk_col3 = st.columns(3)


with risk_col1:

    new_device_values = get_unique_values(
        "new_device"
    )


    if not new_device_values:
        new_device_values = ["No", "Yes"]


    new_device = st.selectbox(
        "New Device",
        new_device_values
    )


with risk_col2:

    location_values = get_unique_values(
        "location_mismatch"
    )


    if not location_values:
        location_values = ["No", "Yes"]


    location_mismatch = st.selectbox(
        "Location Mismatch",
        location_values
    )


with risk_col3:

    ip_risk_score = st.number_input(
        "IP Risk Score",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.01
    )


st.divider()


st.subheader("Customer Profile")


profile_col1, profile_col2, profile_col3, profile_col4 = st.columns(4)


with profile_col1:

    st.info(
        f"**Customer ID**\n\n{selected_customer}"
    )


with profile_col2:

    st.info(
        f"**Home Country**\n\n{home_country}"
    )


with profile_col3:

    st.info(
        f"**KYC Tier**\n\n{kyc_tier}"
    )


with profile_col4:

    st.info(
        f"**Account Age**\n\n{account_age} days"
    )


st.divider()


st.subheader("Fraud Prediction")


predict_button = st.button(
    "🔍 Analyse Transaction",
    type="primary",
    use_container_width=True
)


if predict_button:

    transaction_timestamp = datetime.combine(
        timestamp,
        datetime.min.time()
    ).isoformat()


    payload = {
        "timestamp": transaction_timestamp,
        "customer_id": str(selected_customer),
        "home_country": str(home_country),
        "source_currency": str(source_currency),
        "dest_currency": str(dest_currency),
        "channel": str(channel),
        "amount_sc": float(amount_sc),
        "fee": float(fee),
        "new_device": str(new_device),
        "ip_country": str(ip_country),
        "location_mismatch": str(location_mismatch),
        "ip_risk_score": float(ip_risk_score),
        "kyc_tier": str(kyc_tier),
        "account_age_days": int(account_age),
        "device_trust_score": float(device_trust),
        "risk_score_internal": float(internal_risk),
        "corridor_risk": float(corridor_risk)
    }


    try:

        with st.spinner(
            "Analysing transaction..."
        ):

            response = requests.post(
                API_URL,
                json=payload,
                timeout=30
            )


        if response.status_code == 200:

            result = response.json()


            st.success(
                "Transaction analysis completed successfully."
            )


            st.subheader("Prediction Result")


            result_col1, result_col2, result_col3 = st.columns(3)


            with result_col1:

                if result["is_fraud"] == 1:

                    st.error(
                        "⚠️ Potentially Fraudulent"
                    )

                else:

                    st.success(
                        "✅ Legitimate"
                    )


            with result_col2:

                probability = result[
                    "fraud_probability"
                ]

                st.metric(
                    "Fraud Probability",
                    f"{probability:.2%}"
                )


            with result_col3:

                amount_usd = result.get(
                    "amount_usd",
                    0
                )

                st.metric(
                    "Amount (USD)",
                    f"${amount_usd:,.2f}"
                )


            st.subheader(
                "Transaction Behaviour"
            )


            velocity_col1, velocity_col2, velocity_col3 = st.columns(3)


            with velocity_col1:

                st.metric(
                    "Transactions - Last 1 Hour",
                    result.get(
                        "txn_velocity_1h",
                        0
                    )
                )


            with velocity_col2:

                st.metric(
                    "Transactions - Last 24 Hours",
                    result.get(
                        "txn_velocity_24h",
                        0
                    )
                )


            with velocity_col3:

                st.metric(
                    "Velocity Spike",
                    result.get(
                        "velocity_spike",
                        0
                    )
                )


            with st.expander(
                "View API Request Payload"
            ):

                st.json(payload)


            with st.expander(
                "View API Response"
            ):

                st.json(result)


        else:

            st.error(
                f"API Error {response.status_code}: "
                f"{response.text}"
            )


    except requests.exceptions.ConnectionError:

        st.error(
            "Unable to connect to the FastAPI server. "
            "Please make sure the FastAPI server is running."
        )


    except requests.exceptions.Timeout:

        st.error(
            "The API request timed out. "
            "Please check that the FastAPI server and ML model are running."
        )


    except Exception as e:

        st.error(
            f"An unexpected error occurred: {e}"
        )
