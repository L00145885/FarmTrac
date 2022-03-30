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

cur.execute("CREATE TABLE IF NOT EXISTS cows (cowID INT(15) PRIMARY KEY, breed VARCHAR(15), dob DATE, img BLOB, herdNumber VARCHAR(15), registeredDate DATE, FOREIGN KEY(herdNumber) REFERENCES users(herdNumber))")
db.commit()

cur.execute("CREATE TABLE IF NOT EXISTS procedures (procedureID INT(5) PRIMARY KEY AUTO_INCREMENT , type VARCHAR(15), description VARCHAR (45), dateCompleted DATE, cowID INT(15), FOREIGN KEY(cowID) REFERENCES cows(cowID))")
db.commit()

cur.execute("CREATE TABLE IF NOT EXISTS weights (weightID INT(5) PRIMARY KEY AUTO_INCREMENT , weight INT(5), dateCompleted DATE, cowID INT(15), herdNumber VARCHAR (15), FOREIGN KEY(cowID) REFERENCES cows(cowID), FOREIGN KEY(herdNumber) REFERENCES users(herdNumber))")
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

def findUserWithHerd(herdIn):
    query = "SELECT * FROM users WHERE herdNumber = '"+herdIn+"'"
    cur.execute(query)
    user = cur.fetchone()
    return user

def createCow(cowIDIn, breedIn, dobIn, imgIn, herdIn, dateIn):
        sql = "INSERT INTO cows (cowID, breed, dob, img, herdNumber, registeredDate) VALUES (%s, %s, %s, %s, %s, %s);"
        cur.execute(sql, (cowIDIn, breedIn, dobIn, imgIn, herdIn, dateIn))
        db.commit()
        
def findCow(herdIn):
        query = "SELECT * FROM cows WHERE herdNumber = '"+herdIn+"'"
        cur.execute(query)
        cow = cur.fetchall()
        return cow

def editCow(oldCowIDIn, cowIDIn, breedIn, dobIn):
        sql = """UPDATE cows SET cowID=%s, breed=%s, dob=%s WHERE cowID = %s"""
        cur.execute(sql, (cowIDIn, breedIn, dobIn, oldCowIDIn))
        db.commit()

def createProcedure(dataIn):
        for row in dataIn:
                sql = "INSERT INTO procedures (type, description, dateCompleted, cowID) VALUES (%s, %s, %s, %s);"
                cur.execute(sql, (row[2], row[0], row[1], row[3]))
                db.commit()

def editProcedures(dataIn):
        for row in dataIn:
                sql = """UPDATE procedures SET type=%s, description =%s, dateCompleted=%s WHERE procedureID = %s"""
                cur.execute(sql, (row[1], row[2], row[3], row[0]))
                db.commit()

def deleteCowProcedure(procedureID):
        sql = "DELETE FROM procedures WHERE procedureID = "+procedureID+""
        cur.execute(sql)
        db.commit()

def returnProcedures(cowID):
        query = "SELECT * FROM procedures WHERE cowID = "+cowID+""
        cur.execute(query)
        procedures = cur.fetchall()
        return procedures

def insertWeight(dataIn):
        for row in dataIn:
                sql = "INSERT INTO weights (weight, dateCompleted, cowID, herdNumber) VALUES (%s, %s, %s, %s);"
                cur.execute(sql, (row[0], row[1], row[2], row[3]))
                db.commit() 

def editCowWeights(dataIn):
        for row in dataIn:
                sql = """UPDATE weights SET weight=%s, dateCompleted=%s WHERE weightID = %s"""
                cur.execute(sql, (row[1], row[2], row[0]))
                db.commit()

def deleteCowWeight(weightIDIn):
        sql = "DELETE FROM weights WHERE weightID = "+weightIDIn+""
        cur.execute(sql)
        db.commit()

def returnWeights(cowID):
        query = "SELECT * FROM weights WHERE cowID = "+cowID+""
        cur.execute(query)
        weights = cur.fetchall()
        return weights

def returnWeightsFromHerd(herdID):
        query = "SELECT * FROM weights WHERE herdNumber = "+herdID+""
        cur.execute(query)
        weights = cur.fetchall()
        return weights
                
#def totalNumberOfCows(dateIn):
        #sql = "SELECT * FROM cows WHERE registeredDate <= '"+str(dateIn) +"';"
        #cur.execute(sql)
        #rows = cur.fetchall()
        #return len(rows)