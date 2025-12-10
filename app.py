from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer
from  chromadb import PersistentClient
from chromadb.config import Settings
import requests

#_______________________________________________________
app = Flask(__name__)
conversation = [] #Guarda preguntas y respuestas

OLLAMA_URL = "http://192.168.0.70:11434/api/generate"
MODEL_NAME = "llama3.2"
#_______________________________________________________
#Base de Datos
#incializar ChromaDb Local
cliente = PersistentClient(path="./chroma")
collection = cliente.get_or_create_collection(name="documentos")

#Cargar el modelo de embedding (una sola vez)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def cargar_text_y_embedding():
    if collection.count() > 0:
        return
    with open("info2.txt","r", encoding="utf-8") as f:
        raw_text = f.read()

    fragmentos = [p.strip() for p in raw_text.split("\n") if p.strip]
    embeddings = embedder.encode(fragmentos).tolist()

    for i,fragmento in enumerate(fragmentos):
        collection.add(documents=[fragmento], ids=[f"frag{i}"],embeddings=[embeddings[i]])

cargar_text_y_embedding()

#_______________________________________________________
@app.route("/", methods = ["GET","POST"])
#consulta del usurio
def index():
    global conversation

    if request.method =="POST":
        user_input = request.form["user_input"]

        #Embedding de la pregunta
        embeddin_input= embedder.encode(user_input).tolist()

        #Buscar texto mas similar en la coleccion
        resultados= collection.query(query_embeddings=[embeddin_input], n_results=20) #entregara hasta 10 resultados
        fragmentos = resultados["documents"][0] #se divide en fragmentos
        contexto = "\n".join(fragmentos)

        prompt = f"""
        Ers un asistente amigable y profesional de Cesar.
        
        Cuando el usuario te pregunte por cursos, responde de manera clara y natrual, como sí conversaras con él.

        Agrupa los cursos por tema si es posible, y no repitas el titulo exacto si puedes resumirlo. Usa un tono cercano, como si recomendaras personalmente


        Puedes analizar, contar y filtrar cursos si se te pide. No inventes cursos que no estén presentes en el texto.
        Sí no encuentras lo que se te pregunta, responde exactamente con : "No tengo datos sobre eso."

        Lista de Cursos Disponibles:
        \"\"\"{contexto}\"\"\"

        Pregunta:{user_input} 
        """

        #contexto = cargar_contenido()

        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False         
        }

        response = requests.post(OLLAMA_URL, json=payload)
        result  = response.json()["response"]

        conversation.append(("Tu", user_input))
        conversation.append(("IA",result))
    
    return render_template("index.html", conversation=conversation)

"""
#lectura de contenido 
def cargar_contenido():
    texto1 = open("info1.txt","r",encoding="utf-8").read()
    texto2 = open("info2.txt","r",encoding="utf-8").read()
    
    #el maximo de caracteres es de 8000 caracteres
    return f"{texto1}\n{texto2}"
"""

if __name__=="__main__":
    app.run(debug=True)
 

#_______________________________________________________