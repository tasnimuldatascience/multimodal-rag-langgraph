import streamlit as st
from graphs.agent_graph import create_agent_graph

st.title("Multimodal RAG Agent with LangGraph")

modality = st.selectbox("Select Modality", ["text", "image"])

query_text = None
query_image = None

if modality == "text":
    query_text = st.text_area("Query:")
elif modality == "image":
    uploaded_file = st.file_uploader("Upload image", ["jpg", "png", "jpeg"])
    query_text = st.text_area("Enter your query about the image:")
    
    if uploaded_file:
        query_image = f"temp_{uploaded_file.name}"
        with open(query_image, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.image(query_image, caption="Uploaded Image")

if st.button("Generate Response"):
    if modality == "text" and query_text:
        inputs = {"query": query_text, "modality": "text"}
    elif modality == "image" and query_image and query_text:
        # Here, pass image path as query for embedding, 
        # and the query_text to your generation agent.
        inputs = {"query": query_image, "modality": "image", "text_query": query_text}
    else:
        st.warning("Please provide both an image and a text query.")
        inputs = None

    if inputs:
        graph = create_agent_graph()
        result = graph.invoke(inputs)

        st.subheader("Retrieved Context")
        st.write(result.get('context', 'No context retrieved.'))

        st.subheader("Generated Answer")
        st.write(result.get('response', 'No response generated.'))
