#!/usr/bin/python -tt
import numpy as np
import pandas as pd
from streamlit.server.server import Server
import streamlit as st
import pickle
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from threading import Thread, current_thread
from time import sleep
from subprocess import call, DEVNULL
from sources import data_preparations
import sys


class StreamlitApp:

	def __init__(self):
		self.fighters = pd.read_csv('Data/UFC_Fighters_Database.csv')
		self.fighters2 = pd.read_csv('Data/fighters.csv')
		self.df2 = pd.read_csv('Data/prediction_data.csv', index_col="Unnamed: 0")

		# Switch to this for faster loading.
		mlp_file = open('Models/predict_winner.pkl', 'rb')
		self.mlp_model = pickle.load(mlp_file)
		mlp_file.close()

	def reconstruct(self):
		self.fighters = pd.read_csv('Data/UFC_Fighters_Database.csv')
		self.fighters2 = pd.read_csv('Data/fighters.csv')
		self.df2 = pd.read_csv('Data/prediction_data.csv', index_col="Unnamed: 0")

	def load_models(self):
		return self.mlp_model

	def predict(self, df):
		model = self.load_models()
		prediction = model.predict_proba(df.reshape(1, -1))
		pred = model.predict(df.reshape(1, -1))
		return prediction, pred

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

	def get_headers():
		# Hack to get the session object from Streamlit.

		current_server = Server.get_current()
		if hasattr(current_server, '_session_infos'):
			# Streamlit < 0.56
			session_infos = Server.get_current()._session_infos.values()
		else:
			session_infos = Server.get_current()._session_info_by_id.values()

		# Multiple Session Objects?
		for session_info in session_infos:
			headers = session_info.ws.request.headers
			st.write(headers)

		return headers

	def construct_app(self):
		st.set_page_config(
	        page_title="Predictor",
	        page_icon="chart_with_upwards_trend",
	        layout="wide"
	    )
		st.markdown(
			'<h1 class="header-style" style="text-align: center;"> UFC Predictor </h1>',
			unsafe_allow_html=True
		)
		col1, col2 = st.columns(2)
		# original = Image.open(image)

		# fighter1_img = Image.open("static/fighter_left.png")
		# fighter2_img = Image.open("static/fighter_right.png")

		# col1.image(fighter1_img, caption="Favourite")
		# col2.image(fighter2_img, caption="Underdog")

		# Add all Important Fields

		fighter1 = col1.selectbox(
			'Select Your Favourite Fighter',
			self.fighters2['name']
		)

		fighter2 = col2.selectbox(
			'Select The Underdog Fighter',
			self.fighters2['name']
		)

		## weight class
		fighter1_w = st.selectbox(
			'Weight Class',
			self.df2['weight_class'].unique()
		)

		# fighter2_w = col2.selectbox(
		# 	'Weight Class for Underdog',
		# 	self.df2['weight_class'].unique()
		# )

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

		fighter_t = st.checkbox(
			label="Is the match a Title Match?",
		)

		col1, col2, col3 = st.columns(3)

		data1 = self.fighters2[self.fighters2.name == fighter1]
		data2 = self.fighters2[self.fighters2.name == fighter2]


		# col1.pyplot(graph)
		# col3.write(self.fighters)

		if col2.button('Predict'):
			if fighter1_odd > fighter2_odd:
				st.error("Favourite odd should be less than Underdog!")
			else:
				st.markdown(
					'<h1 class="header-style" style="text-align: center;"> Prediction Results </h1>',
					unsafe_allow_html=True
				)
				data1 = self.fighters2[self.fighters2.name == fighter1]
				data2 = self.fighters2[self.fighters2.name == fighter2]

				columns = ['REACH_delta','SLPM_delta','SAPM_delta','STRA_delta','STRD_delta','TD_delta','TDA_delta','TDD_delta','SUBA_delta','Odds_delta']
				best_cols = ['SLPM_delta', 'SAPM_delta', 'STRD_delta', 'TDD_delta', 'SUBA_delta', 'Odds_delta']

				df = pd.DataFrame(
						[
							[float(data1.reach)-float(data2.reach),
								float(data1.SLpM)-float(data2.SLpM),
								float(data1.SApM)-float(data2.SApM),
								float(data1.Str_Acc)-float(data2.Str_Acc),
								float(data1.Str_Def)-float(data2.Str_Def),
								float(data1.TD_Avg)-float(data2.TD_Avg),
								float(data1.TD_Acc)-float(data2.TD_Acc),
								float(data1.TD_Def)-float(data2.TD_Def),
								float(data1.Sub_Avg)-float(data2.Sub_Avg),
								float(fighter1_odd)-float(fighter2_odd)]
						],
						columns=columns
					)

				# st.dataframe(df)

				pred, d_pred = self.predict(np.array(df[best_cols]))

				# st.write(pred)
				col1, col2 = st.columns(2)

				pred_winner = "Underdog" if d_pred[0] == 0 else "Favourite"

				title_match = 0 if fighter_t == False else 1

				# st.write(pred_winner)
				weight_classes = ['weight_class_Bantamweight','weight_class_Catch Weight', 'weight_class_Featherweight','weight_class_Flyweight', 'weight_class_Heavyweight','weight_class_Light Heavyweight', 'weight_class_Lightweight','weight_class_Middleweight', 'weight_class_Open Weight','weight_class_Super Heavyweight', 'weight_class_Welterweight',"weight_class_Women's Bantamweight","weight_class_Women's Featherweight", "weight_class_Women's Flyweight","weight_class_Women's Strawweight"]
				# fighter1_w
				res = []
				for x in weight_classes:
					if fighter1_w == x.split('_')[2]:
						res.append(1)
					else:
						res.append(0)

				m_cols = ['winner', 'title_fight', 'SLPM_delta', 'SAPM_delta','STRD_delta', 'TDD_delta', 'SUBA_delta', 'weight_class_Bantamweight','weight_class_Catch Weight', 'weight_class_Featherweight','weight_class_Flyweight', 'weight_class_Heavyweight','weight_class_Light Heavyweight', 'weight_class_Lightweight','weight_class_Middleweight', 'weight_class_Open Weight','weight_class_Super Heavyweight', 'weight_class_Welterweight',"weight_class_Women's Bantamweight","weight_class_Women's Featherweight", "weight_class_Women's Flyweight","weight_class_Women's Strawweight"]
				fin = [d_pred[0],
							title_match,
							float(data1.SLpM)-float(data2.SLpM),
							float(data1.SApM)-float(data2.SApM),
							float(data1.Str_Def)-float(data2.Str_Def),
							float(data1.TD_Def)-float(data2.TD_Def),
							float(data1.Sub_Avg)-float(data2.Sub_Avg),
						]
				res = np.array(res)
				# st.write(res)
				fin = np.array(fin)
				dt = np.append(fin,res).reshape(1,22)
				# st.write(dt)

				map_method = {
				    0: "Knock Out or Total Knock Out",
				    1: "UNANIMOUS DECISION",
				    2: "Submission",
				    3: "Disqualification"
				}

				df2 = pd.DataFrame(
						dt,
						columns=m_cols
					)

				mlp_file = open('Models/predict_method.pkl', 'rb')
				mlp_model = pickle.load(mlp_file)
				mlp_file.close()

				pred2 = mlp_model.predict(df2)

				# st.write(pred2[0])

				#####
				### Predicting End round
				#####
				e_cols = ['winner', 'title_fight', 'method','SLPM_delta', 'SAPM_delta','STRD_delta', 'TDD_delta', 'SUBA_delta', 'weight_class_Bantamweight','weight_class_Catch Weight', 'weight_class_Featherweight','weight_class_Flyweight', 'weight_class_Heavyweight','weight_class_Light Heavyweight', 'weight_class_Lightweight','weight_class_Middleweight', 'weight_class_Open Weight','weight_class_Super Heavyweight', 'weight_class_Welterweight',"weight_class_Women's Bantamweight","weight_class_Women's Featherweight", "weight_class_Women's Flyweight","weight_class_Women's Strawweight"]
				fin = [d_pred[0],
							title_match,
							pred2[0],
							float(data1.SLpM)-float(data2.SLpM),
							float(data1.SApM)-float(data2.SApM),
							float(data1.Str_Def)-float(data2.Str_Def),
							float(data1.TD_Def)-float(data2.TD_Def),
							float(data1.Sub_Avg)-float(data2.Sub_Avg),
						]
				res = np.array(res)
				# st.write(res)
				fin = np.array(fin)
				dt = np.append(fin,res).reshape(1,23)

				df3 = pd.DataFrame(
						dt,
						columns=e_cols
					)

				mlp_file = open('Models/predict_end_round.pkl', 'rb')
				mlp_model = pickle.load(mlp_file)
				mlp_file.close()

				pred3 = mlp_model.predict(df3)

				# st.write(pred3[0])

				col1.markdown(
					f'<h3 class="header-style" style="text-align: center; color:red;"> Favourite: {round((pred[0][1]) * 100, 2)}% </h3>',
					unsafe_allow_html=True
				)

				col2.markdown(
					f'<h3 class="header-style" style="text-align: center; color: blue;"> Underdog: {round((pred[0][0]) * 100, 2)}% </h3>',
					unsafe_allow_html=True
				)

				st.markdown(
					f'<h3 class="header-style" style="text-align: center; color: green;"> Break Down Analysis </h3>',
					unsafe_allow_html=True
				)

				st.markdown(
					f'<p style="text-align: center; color: green;">The <strong>{pred_winner}</strong> will most likely win by <strong>{map_method[pred2[0]]}</strong> in round <strong>{pred3[0]}</strong>.<br/> <b>Note</b> that this result is just based on speculation and you shouldn"t bet your money based on this alone.</p>',
					unsafe_allow_html=True
				)

		return self


def data_refresher_function(arg):
	global sa
	t = current_thread()
	# Thread is alive by default
	t.alive = True

	while True:
		if not t.alive:
			break
		
		print("UPDATING FIGHTERS\n..............................................................")
		call(["python", "ufcscraper.py"], stdout=DEVNULL, stderr=DEVNULL)
		print("COMPLETED\n..............................................................")

		print("PREPARING DATA\n..............................................................")
		call(["python", "sources/data_preparations.py"], stdout=DEVNULL, stderr=DEVNULL)
		print("COMPLETED\n..............................................................")

		print("UPDATING STREAMLIT APP\n.................................................")
		# sa.reconstruct()
		# sa.construct_app()
		# Wait for 12hours
		# sleep(43200)

data_refresher = Thread(target = data_refresher_function, args = (10, ))
data_refresher.daemon = True

try:
	# data_refresher.start()

	sa = StreamlitApp()
	sa.construct_app()
	
	# data_refresher.join()
except Exception as e:
	data_refresher.alive = False
	# sys.exit(e)
	print(e)