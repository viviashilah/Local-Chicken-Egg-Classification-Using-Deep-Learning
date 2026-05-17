import datetime
import json
import os
from datetime import datetime
import sqlite3
import logging
from werkzeug.utils import secure_filename
from flask import Flask, jsonify, make_response, render_template, request, send_from_directory
from flask_cors import CORS
import resultmlnew
from dotenv import load_dotenv
import json
import numpy as np
from werkzeug.security import generate_password_hash, check_password_hash
from flask_jwt_extended import create_access_token, get_jwt, get_jwt_identity, jwt_required, JWTManager, set_access_cookies, unset_jwt_cookies
# Konfigurasi logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

load_dotenv()

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)
app.config['JWT_SECRET_KEY'] = os.getenv("JWT_SECRET_KEY", "") 
app.config['JWT_TOKEN_LOCATION'] = ['cookies'] 
app.config["JWT_COOKIE_NAME"] = "access_token_cookie"  # Pastikan sama dengan nama cookie
app.config["JWT_COOKIE_CSRF_PROTECT"] = False  # Matikan CSRF jika tidak digunakan
app.config['JWT_COOKIE_SECURE'] = False  
app.config['JWT_COOKIE_HTTPONLY'] = True  
app.config['JWT_COOKIE_SAMESITE'] = "Lax"  
jwt = JWTManager(app)
UPLOAD_FOLDER = 'uploads'
ASSETS_FOLDER = 'assets'
OUTPUT_FOLDER = 'output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ASSETS_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ASSETS_FOLDER'] = ASSETS_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config["PORT"] = os.getenv("PORT", 3000)
app.config["HOST"] = os.getenv("HOST", "0.0.0.0")

def get_db_connection():
    conn = sqlite3.connect("database.db", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('DROP TABLE IF EXISTS klasifikasi_citra')
    cursor.execute('DROP TABLE IF EXISTS user')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS klasifikasi_citra (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        citra TEXT NOT NULL,
        tanggal TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
        Warna TEXT NOT NULL,
        Probabilitas_Warna DECIMAL NOT NULL,
        Probabilitas_Kebersihan DECIMAL NOT NULL,
        Probabilitas_Keretakan DECIMAL NOT NULL,
        Kebersihan TEXT NOT NULL,
        Keretakan TEXT NOT NULL,
        hasil_klasifikasi TEXT NOT NULL,

        RGB TEXT NOT NULL,
        GLCM TEXT NOT NULL
    )
''')

        # Buat tabel user
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT CHECK(role IN ('admin', 'user')) NOT NULL DEFAULT 'user',
            verifikasi BOOLEAN NOT NULL DEFAULT 0  -- 0 = Belum Terverifikasi, 1 = Terverifikasi
        )
    ''')

    hashed_password = generate_password_hash("admin123")  
    cursor.execute("INSERT INTO user (username, email, password, role, verifikasi) VALUES (?, ?, ?, ?, ?)", 
                   ("ADMIN", "admin@mail.com", hashed_password, "admin", 1))  
    conn.commit()
    conn.close()

init_db()

@app.errorhandler(404)
def not_found(error):
    return render_template("404.html"), 404

@jwt.unauthorized_loader
def unauthorized_response(callback):
    return render_template("401.html"), 401

@jwt.invalid_token_loader
def invalid_token_response(callback):
    return render_template("401.html"), 401

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/egg-class")
# @jwt_required()
# @jwt_required(locations=["cookies"])  
def eggclass():
    return render_template("egg_class.html")

@app.route("/user-guide")
# @jwt_required()
def user_guide():
    return render_template("user_guide.html")

@app.route("/proses/<int:id>")
@jwt_required()
def proses_template(id):
    return render_template("proses.html", id=id)


@app.route("/history-report")
@jwt_required()
def history_report():
    claims = get_jwt() 
    if claims.get("role") != "admin":
        return render_template("403.html"), 403  
    
    return render_template("history_report.html")
@app.route("/data-user")
@jwt_required()
def data_user():
    claims = get_jwt() 
    if claims.get("role") != "admin":
        return render_template("403.html"), 403  
    
    return render_template("data_user.html")
@app.route("/login")
def login():
    return render_template("login.html")
@app.route("/register")
def regiter():
    return render_template("Register.html")

# Endpoint Upload dan Klasifikasi
@app.route('/api/klasifikasi', methods=['POST'])
# @jwt_required()
def add_klasifikasi():
    if 'file' not in request.files:
        return jsonify({"error": "File harus diunggah"}), 400

    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({"error": "File tidak ditemukan"}), 400
    # Pastikan hanya menerima file gambar
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "Format file tidak didukung. Gunakan PNG, JPG, atau JPEG"}), 400
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S%f')
    filename = f"{timestamp}.png"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    try:
        # return mutu, warna, kebersihan,keretakan, np.around(float(Probabilitas_Warna), 2), np.around(float(Probabilitas_Kebersihan), 2),np.around(float(Probabilitas_Keretakan), 2)
        mutu,warna,kebersihan,keretakan,Probabilitas_Warna,Probabilitas_Kebersihan,Probabilitas_Keretakan,rgb_feat, glcm_feat = resultmlnew.predict_quality(file_path,timestamp)
        Probabilitas_Warna = float(Probabilitas_Warna)
        Probabilitas_Kebersihan = float(Probabilitas_Kebersihan)
        Probabilitas_Keretakan = float(Probabilitas_Keretakan)
        print(rgb_feat)
        print(glcm_feat)
    except Exception as e:
        logging.error(f"Error klasifikasi: {str(e)}")
        return jsonify({"error": f"Gagal melakukan klasifikasi: {str(e)}"}), 500

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        
        rgb_json = json.dumps(rgb_feat.tolist())
        glcm_json = json.dumps(glcm_feat.tolist())

        # Query dengan tambahan kolom RGB dan GLCM
        cursor.execute(
            '''INSERT INTO klasifikasi_citra 
            (citra, Warna, Probabilitas_Warna, Probabilitas_Kebersihan, Probabilitas_Keretakan, 
            Kebersihan, Keretakan, hasil_klasifikasi, RGB, GLCM) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (
                file_path, warna, Probabilitas_Warna, Probabilitas_Kebersihan,
                Probabilitas_Keretakan, kebersihan, keretakan, mutu,
                rgb_json, glcm_json
            )
        )

        conn.commit()
        inserted_id = cursor.lastrowid

        cursor.execute("SELECT * FROM klasifikasi_citra WHERE id = ?", (inserted_id,))
        inserted_data = cursor.fetchone()
        print(inserted_data)

        if inserted_data:
            inserted_data_dict = {desc[0]: inserted_data[i] for i, desc in enumerate(cursor.description)}
            return jsonify({"message": "Data berhasil ditambahkan", "data": inserted_data_dict}), 201
        else:
            return jsonify({"error": "Data tidak ditemukan setelah insert"}), 500
    except sqlite3.Error as err:
        logging.error(f"Database error: {str(err)}")
        return jsonify({"error": "Gagal menyimpan ke database"}), 500
    finally:
        cursor.close()
        conn.close()

# Endpoint Mendapatkan Semua Data
@app.route('/api/klasifikasi', methods=['GET'])
@jwt_required()
def get_all_klasifikasi():
    conn = get_db_connection()
    data = conn.execute("SELECT * FROM klasifikasi_citra").fetchall()
    conn.close()
    return jsonify([dict(row) for row in data])

# Endpoint Mendapatkan Data Berdasarkan ID
@app.route('/api/klasifikasi/<int:id>', methods=['GET'])
@jwt_required()
def get_klasifikasi(id):
    conn = get_db_connection()
    data = conn.execute("SELECT * FROM klasifikasi_citra WHERE id = ?", (id,)).fetchone()
    conn.close()
    return jsonify(dict(data)) if data else (jsonify({"error": "Data tidak ditemukan"}), 404)

@app.route('/api/proses/<int:id>', methods=['GET'])
@jwt_required()
def get_proses(id):
    conn = get_db_connection()
    data = conn.execute("SELECT * FROM klasifikasi_citra WHERE id = ?", (id,)).fetchone()
    
    if not data:
        conn.close()
        return jsonify({"error": "Data tidak ditemukan"}), 404

    # Ambil path gambar dari database
    file_path = data["citra"]

    if not os.path.exists(file_path):
        conn.close()
        return jsonify({"error": "File gambar tidak ditemukan"}), 404

    try:
        # Proses prediksi kualitas
        mutu, warna, kebersihan = resultmlnew.predict_quality(file_path)

        # Update hasil klasifikasi di database
        cursor = conn.cursor()
        cursor.execute("UPDATE klasifikasi_citra SET Warna = ?, Kebersihan = ?, hasil_klasifikasi = ? WHERE id = ?", 
                       (warna, kebersihan, mutu, id))
        conn.commit()
        conn.close()

        # Mengembalikan hasil prediksi
        return jsonify({
            "id": id,
            "citra": file_path,
            "Warna": warna,
            "Kebersihan": kebersihan,
            "hasil_klasifikasi": mutu
        })

    except Exception as e:
        conn.close()
        return jsonify({"error": f"Gagal melakukan prediksi: {str(e)}"}), 500


# Endpoint Update Data
@app.route('/api/klasifikasi/<int:id>', methods=['PUT'])
@jwt_required()
def update_klasifikasi(id):
    data = request.json
    if not data:
        return jsonify({"error": "Data tidak boleh kosong"}), 400

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE klasifikasi_citra SET citra = ?, hasil_klasifikasi = ? WHERE id = ?", 
                   (data.get("citra"), data.get("hasil_klasifikasi"), id))
    conn.commit()
    conn.close()
    return jsonify({"message": "Data berhasil diperbarui"}) if cursor.rowcount else (jsonify({"error": "Data tidak ditemukan"}), 404)

# Endpoint Hapus Data
@app.route('/api/klasifikasi/<int:id>', methods=['DELETE'])
@jwt_required()
def delete_klasifikasi(id):
    conn = get_db_connection()
    cursor = conn.cursor()
    data = cursor.execute("SELECT citra FROM klasifikasi_citra WHERE id = ?", (id,)).fetchone()
    if not data:
        conn.close()
        return jsonify({"error": "Data tidak ditemukan"}), 404

    file_path = data["citra"]
    cursor.execute("DELETE FROM klasifikasi_citra WHERE id = ?", (id,))
    conn.commit()
    conn.close()

    if file_path and os.path.exists(file_path):
        os.remove(file_path)

    return jsonify({"message": "Data dan file berhasil dihapus"})

# 📌 CREATE - Tambah User Baru
@app.route('/api/users', methods=['POST'])
def add_user():
    data = request.json
    if not data or not all(k in data for k in ("username", "email", "password")):
        return jsonify({"error": "Username, email, dan password wajib diisi"}), 400

    hashed_password = generate_password_hash(data["password"])
    
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO user (username, email, password, role, verifikasi) VALUES (?, ?, ?, ?, ?)",
                       (data["username"], data["email"], hashed_password, data.get("role", "user"), 0))
        conn.commit()
        user_id = cursor.lastrowid
    except sqlite3.IntegrityError:
        return jsonify({"error": "Username atau email sudah terdaftar"}), 400
    finally:
        conn.close()
    
    return jsonify({"message": "User berhasil ditambahkan", "id": user_id}), 201

# 📌 READ - Get Semua User
@app.route('/api/users', methods=['GET'])
@jwt_required()
def get_all_users():
    conn = get_db_connection()
    cursor = conn.cursor()

    # Mengambil semua user kecuali yang memiliki role 'admin'
    cursor.execute("SELECT id, username, email, verifikasi FROM user WHERE role != ?", ("admin",))
    users = cursor.fetchall()

    conn.close()

    # Konversi hasil ke dalam format list of dictionaries
    user_list = [{"id": row[0], "username": row[1], "email": row[2], "verifikasi": row[3]} for row in users]

    return jsonify({"users": user_list})


# 📌 READ - Get User by ID
@app.route('/api/users/<int:id>', methods=['GET'])
@jwt_required()
def get_user(id):
    conn = get_db_connection()
    user = conn.execute("SELECT id, username, email, role, verifikasi FROM user WHERE id = ?", (id,)).fetchone()
    conn.close()
    return jsonify(dict(user)) if user else (jsonify({"error": "User tidak ditemukan"}), 404)

# 📌 UPDATE - Perbarui User
@app.route('/api/users/<int:id>', methods=['PUT'])
@jwt_required()
def update_user(id):
    data = request.json
    if not data:
        return jsonify({"error": "Data tidak boleh kosong"}), 400

    conn = get_db_connection()
    cursor = conn.cursor()

    fields = []
    values = []
    if "username" in data:
        fields.append("username = ?")
        values.append(data["username"])
    if "email" in data:
        fields.append("email = ?")
        values.append(data["email"])
    if "password" in data:
        fields.append("password = ?")
        values.append(generate_password_hash(data["password"]))
    if "role" in data:
        fields.append("role = ?")
        values.append(data["role"])
    if "verifikasi" in data:
        fields.append("verifikasi = ?")
        values.append(data["verifikasi"])

    if not fields:
        return jsonify({"error": "Tidak ada data yang diperbarui"}), 400

    values.append(id)
    query = f"UPDATE user SET {', '.join(fields)} WHERE id = ?"
    cursor.execute(query, values)
    conn.commit()
    rows_affected = cursor.rowcount
    conn.close()

    return jsonify({"message": "User berhasil diperbarui"}) if rows_affected else (jsonify({"error": "User tidak ditemukan"}), 404)

# 📌 DELETE - Hapus User
@app.route('/api/users/<int:user_id>', methods=['DELETE'])
@jwt_required()
def delete_user(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM user WHERE id = ?", (user_id,))
    conn.commit()
    rows_affected = cursor.rowcount
    conn.close()

    return jsonify({"message": "User berhasil dihapus"}) if rows_affected else (jsonify({"error": "User tidak ditemukan"}), 404)
@app.route('/api/users/<int:user_id>/verify', methods=['PUT'])
@jwt_required()
def verify_user(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("UPDATE user SET verifikasi = 1 WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()

    return jsonify({"message": "User verified successfully"}), 200
# 📌 LOGIN - Simpan Token di Cookie
@app.route('/api/login', methods=['POST'])
def login_user():
    data = request.json
    if not data or not all(k in data for k in ("email", "password")):
        return jsonify({"error": "Email dan password wajib diisi"}), 400

    conn = get_db_connection()
    user = conn.execute("SELECT * FROM user WHERE email = ?", (data["email"],)).fetchone()
    conn.close()

    if not user or not check_password_hash(user["password"], data["password"]):
        return jsonify({"error": "Email atau password salah"}), 401

    if not user["verifikasi"]:
        return jsonify({"error": "Akun belum terverifikasi"}), 403

    # 🔥 FIX: Gunakan `identity` sebagai integer dan tambahan informasi di `additional_claims`
    access_token = create_access_token(
        identity=str(user["id"]),
        additional_claims={"username": user["username"], "role": user["role"]}
    )
    # 🔥 Simpan Token di Cookie
    resp = make_response(jsonify({"message": "Login berhasil"}))
    resp.set_cookie(
        "access_token_cookie", access_token, httponly=True, samesite="Lax", max_age=86400,domain="localhost",secure=False
    )

    return resp

# 📌 LOGOUT - Hapus Token dari Cookie
@app.route('/api/logout', methods=['POST'])
def logout_user():
    response = make_response(jsonify({"message": "Logout berhasil"}))
    unset_jwt_cookies(response)  # Hapus token dari cookie
    return response

# 📌 PROFILE - Menggunakan Token dari Cookie
@app.route('/api/profile', methods=['GET'])
@jwt_required()
def profile():
    current_user = get_jwt()
    return jsonify({"message": "User data retrieved", "user": current_user})

# Endpoint Akses Gambar
@app.route('/uploads/<filename>', methods=['GET'])
# @jwt_required()
def get_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/assets/<filename>', methods=['GET'])
def get_assets(filename):
    return send_from_directory(app.config['ASSETS_FOLDER'], filename)

@app.route('/output/<filename>', methods=['GET'])
def get_output(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True, port=int(app.config["PORT"]), host=app.config["HOST"])
