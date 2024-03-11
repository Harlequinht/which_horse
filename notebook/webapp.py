import streamlit as st
import pandas as pd
import numpy as np


# Your data loading and cleaning functions go here

# Define the pages
pages = ['About Us', 'Which Horse', 'Performance Metrics', 'Project Challenges', 'Project Outcomes']

# Sidebar navigation
selected_page = st.sidebar.selectbox('Navigation', pages)

# Display content based on the selected page
if selected_page == 'About Us':
    st.markdown('## About Us')

    team_data = [
        {"name": "Harlequin", "image_url": "https://images.app.goo.gl/tgbPQTLpeENoNm1C7", "bio": "Bio for Team Member 1"},
        {"name": "Rob", "image_url": "https://images.app.goo.gl/qDmmGBaYuu6nX4x86", "bio": "Bio for Team Member 2"},
        {"name": "Amanda", "image_url": "https://images.app.goo.gl/1hfzoHpgSHRD7Ysa8", "bio": "Bio for Team Member 3"},
        {"name": "Tomi", "image_url": "https://images.app.goo.gl/RcHH4qpXf5dx1wiC7", "bio": "Bio for Team Member 4"},
        {"name": "David", "image_url": "https://images.app.goo.gl/yeTcJSgaqGa3qJnz8", "bio": "Bio for Team Member 5"},
    ]

    for team_member in team_data:
        st.subheader(team_member["name"])
        st.image(team_member["image_url"], use_column_width=True, caption=f"Image of {team_member['name']}")
        st.text(team_member["bio"])
        st.markdown('---')  # Add a horizontal line between team members

elif selected_page == 'Which Horse':
    st.markdown('## Which Horse')
    st.write('''Horse racing holds significant economic and entertainment value, attracting widespread attention from enthusiasts and bettors.

             \nCreating machine learning models to predict horserace outcomes is crucial for enhancing betting strategies, improving the accuracy of odds, and providing valuable insights to both industry professionals and casual participants, ultimately contributing to a more informed and engaging horse racing experience.
             ''')

    st.markdown('## Model 1')
    @st.cache_data
    def get_line_chart_data():

        return pd.DataFrame(
                np.random.randn(20, 3),
                columns=['a', 'b', 'c']
            )

    df = get_line_chart_data()

    st.line_chart(df)

    st.markdown('## Model 2')
    @st.cache_data
    def get_line_chart_data():

        return pd.DataFrame(
                np.random.randn(20, 3),
                columns=['a', 'b', 'c']
            )

    df = get_line_chart_data()

    st.line_chart(df)

    st.markdown('## Model 3')
    @st.cache_data
    def get_line_chart_data():

        return pd.DataFrame(
                np.random.randn(20, 3),
                columns=['a', 'b', 'c']
            )

    df = get_line_chart_data()

    st.line_chart(df)

    st.markdown('## Model 4')
    @st.cache_data
    def get_line_chart_data():

        return pd.DataFrame(
                np.random.randn(20, 3),
                columns=['a', 'b', 'c']
            )

    df = get_line_chart_data()

    st.line_chart(df)

    # Add more content as needed

elif selected_page == 'Performance Metrics':
    st.markdown('## Performance Metrics')
    st.write('''Comparison of loss functions across all of our models. We could include all sorts
             of pretty graphs if we really wanted ! :)
             ''')
    @st.cache_data
    def get_line_chart_data():

        return pd.DataFrame(
                np.random.randn(20, 3),
                columns=['a', 'b', 'c']
            )

    df = get_line_chart_data()

    st.line_chart(df)
    st.write('''Look! We can add multiple graphs. Big fun, many happy!
             ''')
    st.line_chart(df)
    st.write('''And once more, we can add a 3rd graph!
             ''')
    st.line_chart(df)

    # Add more content as needed

elif selected_page == 'Project Challenges':
    st.markdown('## Project Challenges')
    st.write('''Project challenge #1
             \n Project challenge #2
             \n Project challenge #3
                 ''')
    # Add a placeholder for dynamic scrolling
    st.markdown('<div style="height:1000px;"></div>', unsafe_allow_html=True)

        # Create sections with dynamic headings

        # Rest of your Streamlit app code (e.g., data display, graphs, etc.)
        # ...

elif selected_page == 'Project Outcomes':
    st.markdown('## Project Challenges')
    st.write('''Project outcome #1
             \n Project outcome #2
             \n Project outcome #3
                 ''')
    # Add a placeholder for dynamic scrolling
    st.markdown('<div style="height:1000px;"></div>', unsafe_allow_html=True)

        # Create sections with dynamic headings

        # Rest of your Streamlit app code (e.g., data display, graphs, etc.)
