import mysql.connector

conn = mysql.connector.connect(
        host='testdatabase.cb64tjkuejxz.eu-west-1.rds.amazonaws.com',
        user='L00145885',
        password='enTiNoNv#9447')
cur = conn.cursor()
cur.execute("CREATE DATABASE IF NOT EXISTS FarmTrac")

db = mysql.connector.connect(
        host='testdatabase.cb64tjkuejxz.eu-west-1.rds.amazonaws.com',
        user='L00145885',
        password='enTiNoNv#9447',
        database='FarmTrac')
cur = db.cursor()
cur.execute("CREATE TABLE IF NOT EXISTS users (herdNumber VARCHAR(15) PRIMARY KEY, fullName VARCHAR(40), email VARCHAR(50) unique, password VARCHAR(120))")
db.commit()

cur.execute("CREATE TABLE IF NOT EXISTS cows (cowID INT(15) PRIMARY KEY, breed VARCHAR(15), dob DATE, img BLOB, herdNumber VARCHAR(15), FOREIGN KEY(herdNumber) REFERENCES users(herdNumber))")
db.commit()

def createUser(herdNumberIn, fullNameIn, emailIn, passIn):
    sql = "INSERT INTO users (herdNumber, fullName, email, password) VALUES (%s, %s, %s, %s);"
    cur.execute(sql, (herdNumberIn, fullNameIn, emailIn, passIn))
    db.commit()

def findUser(emailIn):
    query = "SELECT email,herdNumber, password FROM users WHERE email = '"+emailIn+"'"
    cur.execute(query)
    user = cur.fetchone()
    return user

def createCow(cowIDIn, breedIn, dobIn, imgIn, herdIn):
        sql = "INSERT INTO cows (cowID, breed, dob, img, herdNumber) VALUES (%s, %s, %s, %s, %s);"
        cur.execute(sql, (cowIDIn, breedIn, dobIn, imgIn, herdIn))
        db.commit()
def findCow(herdIn):
        query = "SELECT * FROM cows WHERE herdNumber = "+herdIn+""
        cur.execute(query)
        cow = cur.fetchall()
        return cow

def editCow(oldCowIDIn, cowIDIn, breedIn, dobIn, herdIn):
        sql = """UPDATE cows SET cowID=%s, breed=%s, dob=%s WHERE cowID = %s"""
        cur.execute(sql, (cowIDIn, breedIn, dobIn, oldCowIDIn))
        db.commit()