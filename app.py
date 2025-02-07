# From streamlit's website
import streamlit as st

# Creating a bordered container to group UI elements
with st.container(border=True):
	# Displaying a heading
	st.write("# Hello, world!")
	# Embedding raw HTML content (custom styled text)
	st.html("""<p style="color:#e0e0e0;"><big><big>A CSUSB Travel Abroad chatbot is coming <b style="color:#ffffff;">soon...</b></big></big></p>""")
