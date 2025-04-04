# import streamlit as st
# import pandas as pd
# import joblib
# import hashlib
# import os


# #here new code


# import os

# def display_plots():
#     st.title('Visualization Plots')
#     plot_files = os.listdir('images')
    
#     for file in plot_files:
#         if file.endswith('.png'):
#             st.image(os.path.join('images', file))
#         elif file.endswith('.html'):
#             html_link = f'<a href="file://{os.path.abspath(os.path.join("images", file))}" target="_blank">{file}</a>'
#             st.markdown(html_link, unsafe_allow_html=True)

# if st.session_state.get("logged_in"):
#     if st.button("View Plots"):
#         display_plots()
# else:
#     st.warning("Please log in to access the plots.")

# #from here

# # Function to hash passwords
# def hash_password(password):
#     return hashlib.sha256(str.encode(password)).hexdigest()

# # Load user credentials
# def load_users():
#     users_df = pd.read_csv('users.csv')
#     users_df['password'] = users_df['password'].apply(hash_password)
#     return users_df

# # Load models
# scaler = joblib.load('scaler.joblib')
# models = {
#     "Logistic Regression": joblib.load('logistic_regression.joblib'),
#     "Decision Tree": joblib.load('decision_tree.joblib'),
#     "K-Nearest Neighbors": joblib.load('k_nearest_neighbors.joblib'),
#     "Support Vector Machine": joblib.load('support_vector_machine.joblib'),
#     "Random Forest": joblib.load('random_forest.joblib')
# }

# # User authentication
# st.sidebar.header("User Login")
# username = st.sidebar.text_input("Username")
# password = st.sidebar.text_input("Password", type="password")
# login_button = st.sidebar.button("Login")

# users = load_users()
# if 'logged_in' not in st.session_state:
#     st.session_state['logged_in'] = False

# if login_button:
#     if not users[(users['username'] == username) & (users['password'] == hash_password(password))].empty:
#         st.session_state['logged_in'] = True
#         st.sidebar.success("Login successful")
#     else:
#         st.sidebar.error("Invalid username or password")

# if st.session_state['logged_in']:
#     st.sidebar.header("User Input Parameters")
#     def user_input_features():
#         data = {
#             'sttl': st.sidebar.number_input("sttl", min_value=0, max_value=255, value=0),
#             'ct_dst_src_ltm': st.sidebar.number_input("ct_dst_src_ltm", min_value=0, max_value=100, value=0),
#             'tcprtt': st.sidebar.number_input("tcprtt", value=0.0),
#             'sbytes': st.sidebar.number_input("sbytes", min_value=0, max_value=100000, value=0),
#             'dbytes': st.sidebar.number_input("dbytes", min_value=0, max_value=100000, value=0),
#             'ct_srv_dst': st.sidebar.number_input("ct_srv_dst", min_value=0, max_value=100, value=0),
#             'smean': st.sidebar.number_input("smean", value=0.0),
#             'ct_srv_src': st.sidebar.number_input("ct_srv_src", min_value=0, max_value=100, value=0),
#             'dur': st.sidebar.number_input("dur", value=0.0),
#             'rate': st.sidebar.number_input("rate", value=0.0)
#         }
#         return pd.DataFrame(data, index=[0])

#     df = user_input_features()
#     st.subheader('User Input Parameters')
#     st.write(df)

#     df_scaled = scaler.transform(df)
#     model_choice = st.selectbox("Choose the model", list(models.keys()))
    
#     if st.button('Predict'):
#         model = models[model_choice]
#         prediction = model.predict(df_scaled)
#         st.subheader('Prediction')
#         st.markdown(f'<h1 style="color: {"red" if prediction[0] else "green"};">{"Attack" if prediction[0] else "Normal"}</h1>', unsafe_allow_html=True)
# else:
#     st.warning("Please login to access the prediction tool")




# import streamlit as st
# import pandas as pd
# import joblib
# import hashlib
# import os

# # -------------------- SESSION STATE INITIALIZATION --------------------

# # Initialize session state variables if they don't exist
# if "logged_in" not in st.session_state:
#     st.session_state["logged_in"] = False
# if "username" not in st.session_state:
#     st.session_state["username"] = ""

# # -------------------- USER AUTHENTICATION --------------------

# # Function to hash passwords
# def hash_password(password):
#     return hashlib.sha256(password.encode()).hexdigest()

# # Load user credentials from CSV
# def load_users():
#     if os.path.exists("users.csv"):
#         users_df = pd.read_csv("users.csv")
#         users_df["password"] = users_df["password"].apply(hash_password)
#         return users_df
#     else:
#         return pd.DataFrame(columns=["username", "password"])  # Return empty DataFrame if file not found

# # Sidebar Login Section
# st.sidebar.header("User Login")
# username = st.sidebar.text_input("Username")
# password = st.sidebar.text_input("Password", type="password")
# login_button = st.sidebar.button("Login")

# users = load_users()

# if login_button:
#     if users[(users["username"] == username) & (users["password"] == hash_password(password))].empty:
#         st.sidebar.error("Invalid username or password")
#     else:
#         st.session_state["logged_in"] = True
#         st.session_state["username"] = username
#         st.sidebar.success(f"Login successful! Welcome, {username}")

# # -------------------- DISPLAY PLOTS --------------------

# def display_plots():
#     st.title("Visualization Plots")
#     plot_dir = "images"

#     if os.path.exists(plot_dir):
#         plot_files = os.listdir(plot_dir)
#         if plot_files:
#             for file in plot_files:
#                 file_path = os.path.join(plot_dir, file)
#                 if file.endswith(".png"):
#                     st.image(file_path)
#                 elif file.endswith(".html"):
#                     st.markdown(f'<a href="file://{os.path.abspath(file_path)}" target="_blank">{file}</a>', unsafe_allow_html=True)
#         else:
#             st.warning("No plots available in the 'images' folder.")
#     else:
#         st.error("Plot directory not found. Make sure 'images' folder exists.")

# # Display plots only if logged in
# if st.session_state["logged_in"]:
#     if st.button("View Plots"):
#         display_plots()
# else:
#     st.warning("Please log in to access the plots.")

# # -------------------- MACHINE LEARNING MODEL LOADING --------------------

# # Load trained scaler
# scaler_path = "scaler.joblib"
# if os.path.exists(scaler_path):
#     scaler = joblib.load(scaler_path)
# else:
#     st.error("Scaler file not found!")

# # Load ML models
# models = {}
# model_files = {
#     "Logistic Regression": "logistic_regression.joblib",
#     "Decision Tree": "decision_tree.joblib",
#     "K-Nearest Neighbors": "k_nearest_neighbors.joblib",
#     "Support Vector Machine": "support_vector_machine.joblib",
#     "Random Forest": "random_forest.joblib"
# }

# for model_name, model_path in model_files.items():
#     if os.path.exists(model_path):
#         models[model_name] = joblib.load(model_path)
#     else:
#         st.error(f"Model file {model_path} not found!")

# # -------------------- USER INPUT PARAMETERS --------------------

# if st.session_state["logged_in"]:
#     st.sidebar.header("User Input Parameters")

#     # Function to capture user input
#     def user_input_features():
#         data = {
#             "sttl": st.sidebar.number_input("sttl", min_value=0, max_value=255, value=0),
#             "ct_dst_src_ltm": st.sidebar.number_input("ct_dst_src_ltm", min_value=0, max_value=100, value=0),
#             "tcprtt": st.sidebar.number_input("tcprtt", value=0.0),
#             "sbytes": st.sidebar.number_input("sbytes", min_value=0, max_value=100000, value=0),
#             "dbytes": st.sidebar.number_input("dbytes", min_value=0, max_value=100000, value=0),
#             "ct_srv_dst": st.sidebar.number_input("ct_srv_dst", min_value=0, max_value=100, value=0),
#             "smean": st.sidebar.number_input("smean", value=0.0),
#             "ct_srv_src": st.sidebar.number_input("ct_srv_src", min_value=0, max_value=100, value=0),
#             "dur": st.sidebar.number_input("dur", value=0.0),
#             "rate": st.sidebar.number_input("rate", value=0.0)
#         }
#         return pd.DataFrame(data, index=[0])

#     df = user_input_features()

#     st.subheader("User Input Parameters")
#     st.write(df)

#     # -------------------- PREDICTION --------------------

#     df_scaled = scaler.transform(df)
#     model_choice = st.selectbox("Choose the model", list(models.keys()))

#     if st.button("Predict"):
#         model = models[model_choice]
#         prediction = model.predict(df_scaled)

#         st.subheader("Prediction")
#         st.markdown(
#             f'<h1 style="color: {"red" if prediction[0] else "green"};">'
#             f'{"Attack" if prediction[0] else "Normal"}</h1>',
#             unsafe_allow_html=True
#         )
# else:
#     st.warning("Please log in to access the prediction tool.")




# import streamlit as st
# import pandas as pd
# import joblib
# import hashlib
# import os

# # -------------------- SESSION STATE INITIALIZATION --------------------

# # Initialize session state variables if they don't exist
# if "logged_in" not in st.session_state:
#     st.session_state["logged_in"] = False
# if "username" not in st.session_state:
#     st.session_state["username"] = ""

# # -------------------- USER AUTHENTICATION --------------------

# # Function to hash passwords
# def hash_password(password):
#     return hashlib.sha256(password.encode()).hexdigest()

# # Load user credentials from CSV
# def load_users():
#     if os.path.exists("users.csv"):
#         users_df = pd.read_csv("users.csv")
#         users_df["password"] = users_df["password"].apply(hash_password)
#         return users_df
#     else:
#         return pd.DataFrame(columns=["username", "password"])  # Return empty DataFrame if file not found

# # Sidebar Login Section
# st.sidebar.header("User Login")
# username = st.sidebar.text_input("Username")
# password = st.sidebar.text_input("Password", type="password")
# login_button = st.sidebar.button("Login")

# users = load_users()

# if login_button:
#     if users[(users["username"] == username) & (users["password"] == hash_password(password))].empty:
#         st.sidebar.error("Invalid username or password")
#     else:
#         st.session_state["logged_in"] = True
#         st.session_state["username"] = username
#         st.sidebar.success(f"Login successful! Welcome, {username}")

# # -------------------- DISPLAY PLOTS --------------------

# def display_plots():
#     st.title("Visualization Plots")
#     plot_dir = "images"

#     if os.path.exists(plot_dir):
#         plot_files = os.listdir(plot_dir)
#         if plot_files:
#             for file in plot_files:
#                 file_path = os.path.join(plot_dir, file)
#                 if file.endswith(".png"):
#                     st.image(file_path)
#                 elif file.endswith(".html"):
#                     st.markdown(f'<a href="file://{os.path.abspath(file_path)}" target="_blank">{file}</a>', unsafe_allow_html=True)
#         else:
#             st.warning("No plots available in the 'images' folder.")
#     else:
#         st.error("Plot directory not found. Make sure 'images' folder exists.")

# # Display plots only if logged in
# if st.session_state["logged_in"]:
#     if st.button("View Plots"):
#         display_plots()
# else:
#     st.warning("Please log in to access the plots.")

# # -------------------- MACHINE LEARNING MODEL LOADING --------------------

# # Load trained scaler (Supports both scaler.pkl and scaler.joblib)
# scaler = None
# scaler_paths = ["scaler.pkl", "scaler.joblib"]
# for scaler_path in scaler_paths:
#     if os.path.exists(scaler_path):
#         scaler = joblib.load(scaler_path)
#         st.sidebar.success(f"Loaded scaler from {scaler_path}")
#         break

# if scaler is None:
#     st.error("Scaler file not found! Please check that `scaler.pkl` or `scaler.joblib` exists.")

# # Load ML models (Supports both .pkl and .joblib formats)
# models = {}
# model_files = {
#     "Logistic Regression": ["logistic_regression.pkl", "logistic_regression.joblib"],
#     "Decision Tree": ["decision_tree.pkl", "decision_tree.joblib"],
#     "K-Nearest Neighbors": ["k_nearest_neighbors.pkl", "k_nearest_neighbors.joblib"],
#     "Support Vector Machine": ["support_vector_machine.pkl", "support_vector_machine.joblib"],
#     "Random Forest": ["random_forest.pkl", "random_forest.joblib"]
# }

# for model_name, paths in model_files.items():
#     for path in paths:
#         if os.path.exists(path):
#             models[model_name] = joblib.load(path)
#             st.sidebar.success(f"Loaded {model_name} model from {path}")
#             break

# if not models:
#     st.error("No ML models found! Please check that `.pkl` or `.joblib` model files exist.")

# # -------------------- USER INPUT PARAMETERS --------------------

# if st.session_state["logged_in"]:
#     st.sidebar.header("User Input Parameters")

#     # Function to capture user input
#     def user_input_features():
#         data = {
#             "sttl": st.sidebar.number_input("sttl", min_value=0, max_value=255, value=0),
#             "ct_dst_src_ltm": st.sidebar.number_input("ct_dst_src_ltm", min_value=0, max_value=100, value=0),
#             "tcprtt": st.sidebar.number_input("tcprtt", value=0.0),
#             "sbytes": st.sidebar.number_input("sbytes", min_value=0, max_value=100000, value=0),
#             "dbytes": st.sidebar.number_input("dbytes", min_value=0, max_value=100000, value=0),
#             "ct_srv_dst": st.sidebar.number_input("ct_srv_dst", min_value=0, max_value=100, value=0),
#             "smean": st.sidebar.number_input("smean", value=0.0),
#             "ct_srv_src": st.sidebar.number_input("ct_srv_src", min_value=0, max_value=100, value=0),
#             "dur": st.sidebar.number_input("dur", value=0.0),
#             "rate": st.sidebar.number_input("rate", value=0.0)
#         }
#         return pd.DataFrame(data, index=[0])

#     df = user_input_features()

#     st.subheader("User Input Parameters")
#     st.write(df)

#     # -------------------- PREDICTION --------------------

#     if scaler is not None:
#         df_scaled = scaler.transform(df)
#         model_choice = st.selectbox("Choose the model", list(models.keys()))

#         if st.button("Predict"):
#             if model_choice in models:
#                 model = models[model_choice]
#                 prediction = model.predict(df_scaled)

#                 st.subheader("Prediction")
#                 st.markdown(
#                     f'<h1 style="color: {"red" if prediction[0] else "green"};">'
#                     f'{"Attack" if prediction[0] else "Normal"}</h1>',
#                     unsafe_allow_html=True
#                 )
#             else:
#                 st.error("Selected model not found! Please train the model and try again.")
#     else:
#         st.error("Scaler is not loaded. Prediction cannot proceed.")
# else:
#     st.warning("Please log in to access the prediction tool.")



# import streamlit as st
# import pandas as pd
# import joblib
# import hashlib
# import os

# # -------------------- SESSION STATE INITIALIZATION --------------------

# # Initialize session state variables if they don't exist
# if "logged_in" not in st.session_state:
#     st.session_state["logged_in"] = False
# if "username" not in st.session_state:
#     st.session_state["username"] = ""

# # -------------------- USER AUTHENTICATION --------------------

# # Function to hash passwords
# def hash_password(password):
#     return hashlib.sha256(password.encode()).hexdigest()

# # Load user credentials from CSV
# def load_users():
#     if os.path.exists("users.csv"):
#         users_df = pd.read_csv("users.csv")
#         users_df["password"] = users_df["password"].apply(hash_password)
#         return users_df
#     else:
#         return pd.DataFrame(columns=["username", "password"])  # Return empty DataFrame if file not found

# # Sidebar Login Section
# st.sidebar.header("User Login")
# username = st.sidebar.text_input("Username")
# password = st.sidebar.text_input("Password", type="password")
# login_button = st.sidebar.button("Login")

# users = load_users()

# if login_button:
#     if users[(users["username"] == username) & (users["password"] == hash_password(password))].empty:
#         st.sidebar.error("Invalid username or password")
#     else:
#         st.session_state["logged_in"] = True
#         st.session_state["username"] = username
#         st.sidebar.success(f"Login successful! Welcome, {username}")

# # -------------------- DISPLAY PLOTS --------------------

# def display_plots():
#     st.title("Visualization Plots")
#     plot_dir = "images"

#     if os.path.exists(plot_dir):
#         plot_files = os.listdir(plot_dir)
#         if plot_files:
#             for file in plot_files:
#                 file_path = os.path.join(plot_dir, file)
#                 if file.endswith(".png"):
#                     st.image(file_path)
#                 elif file.endswith(".html"):
#                     st.markdown(f'<a href="file://{os.path.abspath(file_path)}" target="_blank">{file}</a>', unsafe_allow_html=True)
#         else:
#             st.warning("No plots available in the 'images' folder.")
#     else:
#         st.error("Plot directory not found. Make sure 'images' folder exists.")

# # Display plots only if logged in
# if st.session_state["logged_in"]:
#     if st.button("View Plots"):
#         display_plots()
# else:
#     st.warning("Please log in to access the plots.")

# # -------------------- MACHINE LEARNING MODEL LOADING --------------------

# # Load trained scaler (Supports both scaler.pkl and scaler.joblib)
# scaler = None
# scaler_paths = ["models/scaler.pkl", "models/scaler.joblib"]
# for scaler_path in scaler_paths:
#     if os.path.exists(scaler_path):
#         scaler = joblib.load(scaler_path)
#         st.sidebar.success(f"Loaded scaler from {scaler_path}")
#         break

# if scaler is None:
#     st.error("Scaler file not found! Please check that `scaler.pkl` or `scaler.joblib` exists.")

# # Load ML models (Supports both .pkl and .joblib formats)
# models = {}
# model_files = {
#     "Logistic Regression": ["models/logistic_regression.pkl", "models/logistic_regression.joblib"],
#     "Decision Tree": ["models/decision_tree.pkl", "models/decision_tree.joblib"],
#     "K-Nearest Neighbors": ["models/k_nearest_neighbors.pkl", "models/k_nearest_neighbors.joblib"],
#     "Support Vector Machine": ["models/support_vector_machine.pkl", "models/support_vector_machine.joblib"],
#     "Random Forest": ["models/random_forest.pkl", "models/random_forest.joblib"]
# }

# for model_name, paths in model_files.items():
#     for path in paths:
#         if os.path.exists(path):
#             models[model_name] = joblib.load(path)
#             st.sidebar.success(f"Loaded {model_name} model from {path}")
#             break

# if not models:
#     st.error("No ML models found! Please check that `.pkl` or `.joblib` model files exist.")

# # -------------------- USER INPUT PARAMETERS --------------------

# if st.session_state["logged_in"]:
#     st.sidebar.header("User Input Parameters")

#     # Function to capture user input
#     def user_input_features():
#         data = {
#             "sttl": st.sidebar.number_input("sttl", min_value=0, max_value=255, value=0),
#             "ct_dst_src_ltm": st.sidebar.number_input("ct_dst_src_ltm", min_value=0, max_value=100, value=0),
#             "tcprtt": st.sidebar.number_input("tcprtt", value=0.0),
#             "sbytes": st.sidebar.number_input("sbytes", min_value=0, max_value=100000, value=0),
#             "dbytes": st.sidebar.number_input("dbytes", min_value=0, max_value=100000, value=0),
#             "ct_srv_dst": st.sidebar.number_input("ct_srv_dst", min_value=0, max_value=100, value=0),
#             "smean": st.sidebar.number_input("smean", value=0.0),
#             "ct_srv_src": st.sidebar.number_input("ct_srv_src", min_value=0, max_value=100, value=0),
#             "dur": st.sidebar.number_input("dur", value=0.0),
#             "rate": st.sidebar.number_input("rate", value=0.0)
#         }
#         return pd.DataFrame(data, index=[0])

#     df = user_input_features()

#     st.subheader("User Input Parameters")
#     st.write(df)

#     # -------------------- PREDICTION --------------------

#     if scaler is not None:
#         df_scaled = scaler.transform(df)
#         model_choice = st.selectbox("Choose the model", list(models.keys()))

#         if st.button("Predict"):
#             if model_choice in models:
#                 model = models[model_choice]
#                 prediction = model.predict(df_scaled)

#                 st.subheader("Prediction")
#                 st.markdown(
#                     f'<h1 style="color: {"red" if prediction[0] else "green"};">'
#                     f'{"Attack" if prediction[0] else "Normal"}</h1>',
#                     unsafe_allow_html=True
#                 )
#             else:
#                 st.error("Selected model not found! Please train the model and try again.")
#     else:
#         st.error("Scaler is not loaded. Prediction cannot proceed.")
# else:
#     st.warning("Please log in to access the prediction tool.")






import streamlit as st
import pandas as pd
import joblib
import hashlib
import os

# -------------------- SESSION STATE INITIALIZATION --------------------
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "username" not in st.session_state:
    st.session_state["username"] = ""

# -------------------- USER AUTHENTICATION --------------------

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    if os.path.exists("users.csv"):
        users_df = pd.read_csv("users.csv")
        users_df["password"] = users_df["password"].apply(hash_password)
        return users_df
    return pd.DataFrame(columns=["username", "password"])  # Return empty DataFrame if file not found

st.sidebar.header("User Login")
username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")
login_button = st.sidebar.button("Login")

users = load_users()

if login_button:
    if users[(users["username"] == username) & (users["password"] == hash_password(password))].empty:
        st.sidebar.error("Invalid username or password")
    else:
        st.session_state["logged_in"] = True
        st.session_state["username"] = username
        st.sidebar.success(f"Login successful! Welcome, {username}")

# -------------------- DISPLAY PLOTS --------------------

def display_plots():
    st.title("Visualization Plots")
    plot_dir = "images"
    
    if os.path.exists(plot_dir):
        plot_files = os.listdir(plot_dir)
        if plot_files:
            for file in plot_files:
                file_path = os.path.join(plot_dir, file)
                if file.endswith(".png"):
                    st.image(file_path)
                elif file.endswith(".html"):
                    st.markdown(f'<a href="file://{os.path.abspath(file_path)}" target="_blank">{file}</a>', unsafe_allow_html=True)
        else:
            st.warning("No plots available in the 'images' folder.")
    else:
        st.error("Plot directory not found. Make sure 'images' folder exists.")

if st.session_state["logged_in"]:
    if st.button("View Plots"):
        display_plots()
else:
    st.warning("Please log in to access the plots.")

# -------------------- MACHINE LEARNING MODEL LOADING --------------------
scaler_path = "models/scaler.joblib"
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
else:
    st.error("Scaler file not found!")

models = {}
model_files = {
    "Logistic Regression": "models/logistic_regression.joblib",
    "Decision Tree": "models/decision_tree.joblib",
    "K-Nearest Neighbors": "models/knn.joblib",
    "Support Vector Machine": "models/svm.joblib",
    "Random Forest": "models/random_forest.joblib"
}

for model_name, model_path in model_files.items():
    if os.path.exists(model_path):
        models[model_name] = joblib.load(model_path)
    else:
        st.error(f"Model file {model_path} not found!")

# -------------------- USER INPUT PARAMETERS --------------------
if st.session_state["logged_in"]:
    st.sidebar.header("User Input Parameters")

    def user_input_features():
        data = {
            "sttl": st.sidebar.number_input("sttl", min_value=0, max_value=255, value=0),
            "ct_dst_src_ltm": st.sidebar.number_input("ct_dst_src_ltm", min_value=0, max_value=100, value=0),
            "tcprtt": st.sidebar.number_input("tcprtt", value=0.0),
            "sbytes": st.sidebar.number_input("sbytes", min_value=0, max_value=100000, value=0),
            "dbytes": st.sidebar.number_input("dbytes", min_value=0, max_value=100000, value=0),
            "ct_srv_dst": st.sidebar.number_input("ct_srv_dst", min_value=0, max_value=100, value=0),
            "smean": st.sidebar.number_input("smean", value=0.0),
            "ct_srv_src": st.sidebar.number_input("ct_srv_src", min_value=0, max_value=100, value=0),
            "dur": st.sidebar.number_input("dur", value=0.0),
            "rate": st.sidebar.number_input("rate", value=0.0)
        }
        return pd.DataFrame(data, index=[0])

    df = user_input_features()

    st.subheader("User Input Parameters")
    st.write(df)

    # -------------------- PREDICTION --------------------
    df_scaled = scaler.transform(df)
    model_choice = st.selectbox("Choose the model", list(models.keys()))

    if st.button("Predict"):
        model = models[model_choice]
        prediction = model.predict(df_scaled)

        st.subheader("Prediction")
        st.markdown(
            f'<h1 style="color: {"red" if prediction[0] else "green"};">'
            f'{"Attack" if prediction[0] else "Normal"}</h1>',
            unsafe_allow_html=True
        )
else:
    st.warning("Please log in to access the prediction tool.")
