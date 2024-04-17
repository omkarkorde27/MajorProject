from flask import *
from sqlite3 import *
import yake
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

app = Flask(__name__)
app.secret_key = "kamalsir"

@app.route("/home", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        session.pop("un")
        return redirect(url_for("login"))
    if "un" in session:
        return render_template("home.html", m=session["un"])
    else:
        return redirect(url_for("login"))

@app.route("/stuhome", methods=["GET", "POST"])
def stuhome():
    if request.method == "POST":
        session.pop("un")
        return redirect(url_for("login"))
    if "un" in session:
        return render_template("stuhome.html", m=session["un"])
    else:
        return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        un = request.form["un"]
        pw = request.form["pw"]
        conn = None
        try:
            conn = connect("kc.db")
            cursor = conn.cursor()
            sql = "select * from studentslogin where un = '%s' and pw = '%s'"
            cursor.execute(sql % (un, pw))
            data = cursor.fetchall()
            if len(data) == 0:
                return render_template("login.html", m="invalid username/ password ")
            else:
                session["un"] = un
                return redirect(url_for("studashboard"))
        except Exception as e:
            return render_template("signup.html", m="issue" + str(e))
        finally:
            if conn is not None:
                conn.close()
    else:
        return render_template("login.html")

@app.route("/studashboard")
def studashboard():
    if "un" in session:
        return render_template("studashboard.html")  # Render your exam HTML
    else:
        return redirect(url_for("login"))

@app.route('/submit_exam', methods=['POST'])
def submit_exam():
    if "un" in session:
        answer1 = request.form['answer1']
        un = session['un']
        print("Retrieved Answer:", answer1) 
        print("Retrieved Username:", un) 
        conn = connect('kc.db')
        cursor = conn.cursor()

        # Insert the answers and username into the students table
        cursor.execute("INSERT INTO students (answers) VALUES (?)", (answer1,))

        # Commit changes and close the connection
        conn.commit()
        conn.close()

        return redirect(url_for('stuhome'))

    else:
        return redirect(url_for('login'))


@app.route("/professorlogin", methods=["GET", "POST"])
def professorlogin():
    if request.method == "POST":
        un = request.form["un"]
        pw = request.form["pw"]
        conn = None
        try:
            conn = connect("kc.db")
            cursor = conn.cursor()
            sql = "select * from users where un = '%s' and pw = '%s'"
            cursor.execute(sql % (un, pw))
            data = cursor.fetchall()
            if len(data) == 0:
                return render_template("professorlogin.html", m="Invalid access")
            else:
                if data[0][2] != 'professor':  # Check the role from fetched data
                    return render_template("professorlogin.html", m="Unauthorized access")
                else:
                    session["un"] = un
                    return redirect(url_for("prof_dashboard"))
        except Exception as e:
            return render_template("signup.html", m="An issue occurred: " + str(e))
        finally:
            if conn is not None:
                conn.close()
    else:
        return render_template("professorlogin.html")

@app.route('/prof_dashboard')
def prof_dashboard():
    if 'un' in session:  # Assuming the professor is logged in
        # Connect to the SQLite database
        conn = connect('kc.db')
        cursor = conn.cursor()

        # Fetch student answers from the database
        cursor.execute("SELECT un, answers FROM students")
        fetched_data = cursor.fetchall() 

        # Close the database connection
        conn.close()

        # Render a template, passing the fetched data
        return render_template('prof_dashboard.html', student_answers=fetched_data)
    else:
        return redirect(url_for('professorlogin'))

@app.route('/compare_answers', methods=['GET', 'POST'])
def compare_answers():
    model_answer = request.form['model_answer']
    student_answer = request.form['student_answer']

    model = SentenceTransformer('all-MiniLM-L6-v2')
    nli_model = pipeline("text-classification", model="roberta-large-mnli")

    def generate_contextual_embeddings(text):
        # Generate embeddings
        embeddings = model.encode(text)
        return embeddings

    kw_extractor = yake.KeywordExtractor()
    language = "en"
    max_ngram_size = 3
    deduplication_threshold = 0.9
    numOfKeywords = 8
    custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, top=numOfKeywords, features=None)

    keywords = custom_kw_extractor.extract_keywords(model_answer)
    key_phrases = set([kw[0] for kw in keywords])

    keywords.sort()    

    keyword_presence = sum([1 for k in key_phrases if k in student_answer.lower()]) / len(key_phrases)

    student_embeddings = generate_contextual_embeddings(student_answer)
    model_embeddings = generate_contextual_embeddings(model_answer)

    cos_sim = util.cos_sim(student_embeddings, model_embeddings).item()

    def get_nli_score(student_answer, model_answer):
        # Using the model answer as a premise and the student answer as a hypothesis
        nli_result = nli_model(f"{model_answer} [SEP] {student_answer}")
        entailment_score = next((item for item in nli_result if item['label'] == 'ENTAILMENT'), None)
        return entailment_score['score'] if entailment_score else 0

    # Adjust the combined score calculation to include the NLI score
    nli_score = get_nli_score(student_answer, model_answer)

    combined_score = 0.55 * cos_sim + 0.15 * keyword_presence + 0.30 * nli_score   

    if 80 <= combined_score <= 100:
        grade = 'O'
    elif 70 <= combined_score < 80:
        grade = 'A'
    elif 60 <= combined_score < 70:
        grade = 'B'
    elif 50 <= combined_score < 60:
        grade = 'C'
    else:  # combined_score < 50
        grade = 'F'
 
    print(f"Grade: {grade}") 

    return render_template('results.html', model_answer=model_answer, student_answer=student_answer, grade=grade, keywords=keywords, cos_sim=cos_sim, nli_score=nli_score)

@app.route('/examine', methods=['POST']) 
def examine():
    student_answer = request.form.get('answer')  
    return render_template('examine.html', student_answer=student_answer)    

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if "un" in session:
        return redirect(url_for("home"))
    if request.method == "POST":
        un = request.form["un"]
        pw1 = request.form["pw1"]
        pw2 = request.form["pw2"]
        if pw1 == pw2:
            conn = connect("kc.db")
            cursor = conn.cursor()
            try:
                sql = "insert into studentslogin values('%s', '%s')"
                cursor.execute(sql % (un, pw1))
                conn.commit()
                return redirect(url_for("login"))
            except Exception as e:
                conn.rollback()
                return render_template("signup.html", m="user already exists " + str(e))
            finally:
                conn.close()
        else:
            return render_template("signup.html", m="passwords did not match ")
    else:
        return render_template("signup.html")

@app.route('/main')
def main():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True, use_reloader=True)
