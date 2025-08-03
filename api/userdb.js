/* eslint-disable no-unreachable */
import fs from 'fs/promises';
import client from './dbclient.js';
import bcrypt from 'bcryptjs';
async function init_db() {
    try {
        const users = client.db('lab5db').collection('users');
        if ((await users.countDocuments()) == 0) {
            const data = await fs.readFile('./users.json', 'utf-8');
            const userData = JSON.parse(data);
            var num = (await users.insertMany(userData)).insertedCount;
            console.log(`Added ${num} users`);
        }
    } catch (err) {
        console.log("Unable to initialize the database!");
    }
}
async function verifyPassword(password, hashedPassword) {
    const isMatch = await bcrypt.compare(password, hashedPassword);
    return isMatch; 
  }
async function encryptPassword(password) {
    const saltRounds = 10; 
    const hashedPassword = await bcrypt.hash(password, saltRounds);
    return hashedPassword; 
  }
async function validate_user(username,password) {

    if((username==''||password==''))return !(username==''||password=='');
    try {
        const users = client.db('lab5db').collection('users');
        var user = await users.findOne({username:username});
        if(user==null) return false;
        if(user.password!=password) return false;
        return user;
    }catch(err){
        console.log("Unable to fetch from database!");
    }

}
async function update_user(userid,username, password, role, enabled,birthday,email) {
    try {
        const users = client.db('lab5db').collection('users');
        const query = { username: username };
        const update = {
            $set: {
                userid:userid,
                username:username,
                birthday:birthday,
                email:email,
                password: password,
                role: role,
                enabled: enabled
            }
        };
        const options = { upsert: true };
        const result = await users.updateOne(query, update, options);
        if (result.upsertedCount > 0) {
            console.log("Added 1 user");
        } else {
            console.log("Added 0 user");
        }
        return true;
    }catch(err){
        console.log("Unable to update the database!");
    }
}
async function fetch_user(username) {
    try {
        const users = client.db('lab5db').collection('users');
        const query = { username: username };
        const user = await users.findOne(query);
        if(user==null)return false;
        else return user;
    } catch (error) {
        console.log("Unable to fetch from database!");
        return null;
    } 
}
async function fetch_user_update(userid,username) {
    try {
        const users = client.db('lab5db').collection('users');
        const query = { username: username,userid:userid };
        const user = await users.findOne(query);
        console.log("!null");
        if(user==null)return false;
        else return true;
    } catch (error) {
        console.log("Unable to fetch from database!");
        return null;
    } 
}
async function username_exist(username) {
    try {
        const user = await fetch_user(username);
        return user !== null;
    } catch (error) {
        console.log("Unable to fetch from database!");
        return false;
    }
}
// async function finduser_by_id(userid) {
//     try {
//         const users = client.db('lab5db').collection('users');
//         const query = { username: username };
//         const user = await users.findOne(query);
//         console.log("!null");
//         if(user==null)return false;
//         else return user;
//     } catch (error) {
//         console.log("Unable to fetch from database!");
//         return null;
//     } 
// }
async function update_user_byid(userid,username, password,staticEmail,birthday,gender) {
    console.log(gender);
    try {
        const users = client.db('lab5db').collection('users');
        const query = { userid: userid };
        console.log(userid);
        const update = {
            $set: {
                userid: userid,
                password: password,
                username: username,
                staticEmail: staticEmail,
                birthday:birthday,
                gender:gender
            }
        };
        const options = { upsert: false };
        const result = await users.updateOne(query, update, options);
        if (result.upsertedCount > 0) {
            console.log("Added 1 user");
        } else {
            console.log("Added 0 user");
        }
        return true;
    }catch(err){
        console.log("Unable to update the database!");
    }
}


export {init_db, validate_user, update_user, fetch_user, username_exist,update_user_byid,fetch_user_update};