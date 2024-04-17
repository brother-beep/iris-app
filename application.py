import pickle
import streamlit as st  

with open("tree.pkl","rb") as f:
    model = pickle.load(f)
    

def predict_class(sepal_length,sepal_width,petal_length,petal_width):
    values = [sepal_length,sepal_width,petal_length,petal_width]
    prediction = model.predict([values])
    return prediction[0]



def main():
    st.title("Iris Flower Category")
    sepal_length = st.slider("sepal_length",0.0,6.0)
    sepal_width = st.slider("sepal_width",0.0,6.0)
    petal_length = st.slider("petal_length",0.0,6.0)
    petal_width = st.slider("petal_width",0.0,6.0)
    
    if st.button("select"):
        result = predict_class(sepal_length,sepal_width,petal_length,petal_width)
        st.write(result)
    
    
if __name__ == "__main__":
    main()
    