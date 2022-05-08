import mysql.connector

#connect to DB instance
conn = mysql.connector.connect(
        host='testdatabase.cb64tjkuejxz.eu-west-1.rds.amazonaws.com',
        user='L00145885',
        password='enTiNoNv#9447')
cur = conn.cursor()
cur.execute("CREATE DATABASE IF NOT EXISTS FarmTrac") #create database

#connect to DB
db = mysql.connector.connect(
        host='testdatabase.cb64tjkuejxz.eu-west-1.rds.amazonaws.com',
        user='L00145885',
        password='enTiNoNv#9447',
        database='FarmTrac')
cur = db.cursor()
#create tables
cur.execute("CREATE TABLE IF NOT EXISTS users (userID INT(5) AUTO_INCREMENT, herdNumber VARCHAR(15) PRIMARY KEY, fullName VARCHAR(40), email VARCHAR(50) unique, password VARCHAR(120), INDEX(userID))")
db.commit()

cur.execute("CREATE TABLE IF NOT EXISTS cows (cowID INT(15) PRIMARY KEY, breed VARCHAR(15), dob DATE, img BLOB, herdNumber VARCHAR(15), registeredDate DATE, FOREIGN KEY(herdNumber) REFERENCES users(herdNumber) ON DELETE CASCADE ON UPDATE CASCADE)")
db.commit()

cur.execute("CREATE TABLE IF NOT EXISTS procedures (procedureID INT(5) PRIMARY KEY AUTO_INCREMENT , type VARCHAR(15), description VARCHAR (45), dateCompleted DATE, cowID INT(15), FOREIGN KEY(cowID) REFERENCES cows(cowID) ON DELETE CASCADE)")
db.commit()

cur.execute("CREATE TABLE IF NOT EXISTS weights (weightID INT(5) PRIMARY KEY AUTO_INCREMENT , weight INT(5), dateCompleted DATE, cowID INT(15), herdNumber VARCHAR (15), FOREIGN KEY(cowID) REFERENCES cows(cowID) ON DELETE CASCADE, FOREIGN KEY(herdNumber) REFERENCES users(herdNumber) ON DELETE CASCADE ON UPDATE CASCADE)")
db.commit() 

#create new user
def createUser(herdNumberIn, fullNameIn, emailIn, passIn):
    sql = "INSERT INTO users (herdNumber, fullName, email, password) VALUES (%s, %s, %s, %s);"
    cur.execute(sql, (herdNumberIn, fullNameIn, emailIn, passIn))
    db.commit()

#delete user from DB
def deleteUserFromDB(herdNo):
        sql = "DELETE FROM users WHERE herdNumber = %s"
        params = (herdNo, )
        cur.execute(sql, params)
        db.commit()

#update user details
def updateUserInDB(herdNumberIn, fullNameIn, emailIn, userIDIn):
        sql = """UPDATE users SET herdNumber=%s, fullName=%s, email=%s WHERE userID = %s"""
        cur.execute(sql, (herdNumberIn, fullNameIn, emailIn, userIDIn))
        db.commit()

#retrieve specific user with email
def findUser(emailIn):
    query = "SELECT * FROM users WHERE email = %s"
    params = (emailIn,)
    cur.execute(query,params)
    user = cur.fetchone()
    return user

def findUserForSaving(idIn, emailIn):
        query = "SELECT * FROM users WHERE userID != %s AND email = %s"
        params = (idIn,emailIn)
        cur.execute(query,params)
        user = cur.fetchone()
        return user

#retrieve specific user with herd number
def findUserWithHerd(herdIn):
    query = "SELECT * FROM users WHERE herdNumber = %s"
    params = (herdIn,)
    cur.execute(query,params)
    user = cur.fetchone()
    return user

def findUserWithHerdForSaving(idIn, herdIn):
        query = "SELECT * FROM users WHERE userID != %s AND herdNumber = %s"
        params = (idIn,herdIn)
        cur.execute(query,params)
        user = cur.fetchone()
        return user

#insert cow profile to DB
def createCow(cowIDIn, breedIn, dobIn, imgIn, herdIn, dateIn):
        sql = "INSERT INTO cows (cowID, breed, dob, img, herdNumber, registeredDate) VALUES (%s, %s, %s, %s, %s, %s);"
        cur.execute(sql, (cowIDIn, breedIn, dobIn, imgIn, herdIn, dateIn))
        db.commit()

#return all cows in associated herd number     
def findCow(herdIn):
        query = "SELECT * FROM cows WHERE herdNumber = %s"
        params = (herdIn, )
        cur.execute(query, params)
        cow = cur.fetchall()
        return cow

#return details for one specific cow
def findCowWithCowID(cowID):
        sql = "SELECT * FROM cows WHERE cowID = %s"
        params = (cowID, )
        cur.execute(sql, params)
        cow = cur.fetchone()
        return cow

#update record for specific cow
def editCow(oldCowIDIn, cowIDIn, breedIn, dobIn):
        sql = """UPDATE cows SET cowID=%s, breed=%s, dob=%s WHERE cowID = %s"""
        cur.execute(sql, (cowIDIn, breedIn, dobIn, oldCowIDIn))
        db.commit()

#delete cow record from DB
def deleteCowFromDB(cowID):
        sql = "DELETE FROM cows WHERE cowID = %s"
        params = (cowID, )
        cur.execute(sql, params)
        db.commit()

#add medical procedure to DB
def createProcedure(dataIn):
        for row in dataIn:
                sql = "INSERT INTO procedures (type, description, dateCompleted, cowID) VALUES (%s, %s, %s, %s);"
                cur.execute(sql, (row[2], row[0], row[1], row[3]))
                db.commit()

#update medical procedures for one cow
def editProcedures(dataIn):
        for row in dataIn:
                sql = """UPDATE procedures SET type=%s, description =%s, dateCompleted=%s WHERE procedureID = %s"""
                cur.execute(sql, (row[1], row[2], row[3], row[0]))
                db.commit()

#delete specific medical procedure
def deleteCowProcedure(procedureID):
        sql = "DELETE FROM procedures WHERE procedureID = %s"
        params = (procedureID, )
        cur.execute(sql, params)
        db.commit()

#return procedures for one cow
def returnProcedures(cowID):
        query = "SELECT * FROM procedures WHERE cowID = %s ORDER BY dateCompleted"
        params = (cowID, )
        cur.execute(query, params)
        procedures = cur.fetchall()
        return procedures

#insert weights for specific cow
def insertWeight(dataIn):
        for row in dataIn:
                sql = "INSERT INTO weights (weight, dateCompleted, cowID, herdNumber) VALUES (%s, %s, %s, %s);"
                cur.execute(sql, (row[0], row[1], row[2], row[3]))
                db.commit() 

#edit weights for specific cow
def editCowWeights(dataIn):
        for row in dataIn:
                sql = """UPDATE weights SET weight=%s, dateCompleted=%s WHERE weightID = %s"""
                cur.execute(sql, (row[1], row[2], row[0]))
                db.commit()

#delete one weight record for cow
def deleteCowWeight(weightIDIn):
        sql = "DELETE FROM weights WHERE weightID = %s"
        params = (weightIDIn, )
        cur.execute(sql, params)
        db.commit()

#return all weights for one cow
def returnWeights(cowID):
        query = "SELECT * FROM weights WHERE cowID = %s ORDER BY dateCompleted"
        params = (cowID, )
        cur.execute(query, params)
        weights = cur.fetchall()
        return weights

#return all cow weights in entire herd
def returnWeightsFromHerd(herdID):
        query = "SELECT * FROM weights WHERE herdNumber = %s ORDER BY dateCompleted"
        params = (herdID, )
        cur.execute(query, params)
        weights = cur.fetchall()
        return weights