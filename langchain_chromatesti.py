import openai
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
import pandas as pd
import os

# OpenAI API avain
openai.api_key = ''

# Lataa tiedot Excel-tiedostosta ja varmista, että LVI-koodi on merkkijono
tiedot = pd.read_excel("tuotteet.xlsx")[["LVI-koodi", "Yleisnimi", "Tekninen nimi"]]

# Varmista, että LVI-koodi on merkkijono
tiedot["LVI-koodi"] = tiedot["LVI-koodi"].astype(str)

# Langchainin OpenAI upotusmalli
embedding_model = OpenAIEmbeddings(openai_api_key=openai.api_key)

# Chroma-tietovaraston polku
chromadb_polku = "chroma_db_storage"

# Tarkistetaan, onko tietovarasto olemassa
if os.path.exists(chromadb_polku):
    print("Tietovarasto löytyi, ladataan se...")
    vectorstore = Chroma(persist_directory=chromadb_polku, embedding_function=embedding_model)
else:
    print("Tietovarastoa ei löytynyt, luodaan uusi...")
    documents = [
        Document(
            page_content=f"{row['LVI-koodi']} {row['Yleisnimi']} {row['Tekninen nimi']}",
            metadata=row.to_dict()
        )
        for _, row in tiedot.iterrows()
    ]
    vectorstore = Chroma.from_documents(documents, embedding_model, persist_directory=chromadb_polku)
    print("Uusi tietovarasto luotu.")

# Funktio synonyymien ja kirjoitusvirheiden korjaamiseen
def prosessoi_teksti_gpt(kysely):
    try:
        # Käytetään oikeaa OpenAI:n ChatCompletion APIa
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",  # Malli
            messages=[  # Viestit
                {"role": "system", "content": "Olet avustaja, joka osaa jakaa tekstin avainsanoihin, korjata kirjoitusvirheet ja tulkita mahdollisia synonyymejä."},
                {"role": "user", "content": f"Jaa seuraava teksti avainsanoihin ja vertaa niitä annettuun tuotteet.xlsx tiedoston yleisnimiin. \
                Palauta tiedot vain nimikkeiden osalta jotka tekstissä mainitaan edes lähes samassa muodossa tai synonyymina. Ota huomioon myös taivutusmuodot: {kysely}"}
            ]
        )

        # Parsitaan vastaus: oikea tapa käyttää vastausta
        avainsanat = response.choices[0].message.content.strip()

        # Palautetaan lista avainsanoista
        return [s.strip() for s in avainsanat.split(",") if s.strip()]

    except Exception as e:
        print(f"Tuntematon virhe: {e}")
        return [kysely]
    
# Kysely
kysely = "joo tällästä tänää meni viemäröintiputkea eikä oikeestaan muuta"

# Esikäsitellään kysely ja laajennetaan se
avainsanat = prosessoi_teksti_gpt(kysely)
print(f"Avainsanat kyselystä: {avainsanat}")

# Haetaan tulokset jokaiselle avainsanalle
osumat = []
for avainsana in avainsanat:
    try:
        tulokset = vectorstore.similarity_search(avainsana, k=5)
        osumat.extend(tulokset)
    except Exception as e:
        print(f"Virhe haussa avainsanalle '{avainsana}': {e}")

# Tulostetaan osumat
if osumat:
    print(f"Löytyi {len(osumat)} osumaa:")
    for i, osuma in enumerate(osumat, 1):
        print(f"Osuma {i}:")
        print(f"LVI-koodi: {osuma.metadata['LVI-koodi']}")
        print(f"Yleisnimi: {osuma.metadata['Yleisnimi']}")
        print(f"Tekninen nimi: {osuma.metadata['Tekninen nimi']}")
        print("-" * 40)
else:
    print("Ei osumia.")
