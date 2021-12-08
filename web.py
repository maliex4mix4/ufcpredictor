#!/usr/bin/python -tt
import numpy as np
import pandas as pd
import streamlit as st
import pickle
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from sklearn.neural_network import MLPClassifier


class StreamlitApp:

	def __init__(self):
		self.fighters = pd.read_csv('Data/UFC_Fighters_Database.csv')
		self.fighters2 = pd.read_csv('Data/ufc_fighters.csv')

	def load_models(self):
		mlp_file = open('Models/predict_winner.pkl', 'rb')
		mlp_model = pickle.load(mlp_file)
		mlp_file.close()

		return mlp_model

	def predict(self, df):
		model = self.load_models()
		prediction = model.predict_proba(df.reshape(1, -1))
		return prediction

	def construct_sidebar(self):

		cols = [col for col in features.columns]

		st.sidebar.markdown(
			'<p class="header-style">Iris Data Classification</p>',
			unsafe_allow_html=True
		)
		sepal_length = st.sidebar.selectbox(
			f"Select {cols[0]}",
			sorted(features[cols[0]].unique())
		)

		sepal_width = st.sidebar.selectbox(
			f"Select {cols[1]}",
			sorted(features[cols[1]].unique())
		)

		petal_length = st.sidebar.selectbox(
			f"Select {cols[2]}",
			sorted(features[cols[2]].unique())
		)

		petal_width = st.sidebar.selectbox(
			f"Select {cols[3]}",
			sorted(features[cols[3]].unique())
		)
		values = [sepal_length, sepal_width, petal_length, petal_width]

		return values

	def plot_pie_chart(self, probabilities, labels):
		fig = go.Figure(
			data=[go.Pie(
					labels=labels,
					values=probabilities,
					pull=[0, 0.2, 0]
			)]
		)
		fig = fig.update_traces(
			hoverinfo='label+percent',
			textinfo='value',
			textfont_size=15
		)
		return fig

	def construct_app(self):
		st.set_page_config(layout="wide")
		st.markdown(
			'<h1 class="header-style" style="text-align: center;"> UFC Predictor </h1>',
			unsafe_allow_html=True
		)
		col1, col2 = st.columns(2)
		# original = Image.open(image)
		fighter1_img = Image.open("static/fighter_left.png")
		fighter2_img = Image.open("static/fighter_right.png")

		col1.image(fighter1_img, caption="Favourite")
		col2.image(fighter2_img, caption="Underdog")

		# Add all Important Fields

		fighter1 = col1.selectbox(
			'Select Your Favourite Fighter',
			self.fighters['NAME']
		)

		fighter2 = col2.selectbox(
			'Select The Underdog Fighter',
			self.fighters['NAME']
		)

		## weight class
		fighter1_w = col1.selectbox(
			'Weight Class',
			self.fighters['WeightClass'].unique()
		)

		fighter2_w = col2.selectbox(
			'Weight Class for Underdog',
			self.fighters['WeightClass'].unique()
		)

		## Decimal Odds
		fighter2_odd = col2.number_input(
			label="Underdog Decimal Odd",
			min_value=0.00,
			step=0.01,
		)
		fighter1_odd = col1.number_input(
			label="Decimal Odd",
			min_value=0.00,
			step=0.01,
		)

		col1, col2, col3 = st.columns(3)

		data1 = self.fighters2[self.fighters2.name == fighter1]
		data2 = self.fighters2[self.fighters2.name == fighter2]

		graph = self.plot_pie_chart([data1.win, data1.lose, data1.draw], ['Win', "Lose", "Draw"])

		# col1.pyplot(graph)

		if col2.button('Predict'):
			if fighter1_odd > fighter2_odd:
				st.error("Favourite odd should be less than Underdog!")
			else:
				st.markdown(
					'<h1 class="header-style" style="text-align: center;"> Prediction Results </h1>',
					unsafe_allow_html=True
				)
				data1 = self.fighters[self.fighters.NAME == fighter1]
				data2 = self.fighters[self.fighters.NAME == fighter2]

				columns = ['REACH_delta','SLPM_delta','SAPM_delta','STRA_delta','STRD_delta','TD_delta','TDA_delta','TDD_delta','SUBA_delta','Odds_delta']
				best_cols = ['SLPM_delta', 'SAPM_delta', 'STRD_delta', 'TDD_delta', 'SUBA_delta', 'Odds_delta']

				df = pd.DataFrame(
						[
							[float(data1.REACH)-float(data2.REACH),
								float(data1.SLPM)-float(data2.SLPM),
								float(data1.SAPM)-float(data2.SAPM),
								float(data1.STRA)-float(data2.STRA),
								float(data1.STRD)-float(data2.STRD),
								float(data1.TD)-float(data2.TD),
								float(data1.TDA)-float(data2.TDA),
								float(data1.TDD)-float(data2.TDD),
								float(data1.SUBA)-float(data2.SUBA),
								float(fighter1_odd)-float(fighter2_odd)]
						],
						columns=columns
					)

				# st.dataframe(df)

				pred = self.predict(np.array(df[best_cols]))

				# st.write(pred)
				col1, col2 = st.columns(2)

				col1.markdown(
					f'<h3 class="header-style" style="text-align: center; color:red;"> Favourite: {round((pred[0][0]) * 100, 2)}% </h3>',
					unsafe_allow_html=True
				)

				col2.markdown(
					f'<h3 class="header-style" style="text-align: center; color: blue;"> Underdog: {round((pred[0][1]) * 100, 2)}% </h3>',
					unsafe_allow_html=True
				)				

		return self

sa = StreamlitApp()
sa.construct_app()