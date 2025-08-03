/* eslint-disable no-unreachable */
import fs from 'fs/promises';
import client from './dbclient.js';
async function init_db() {
    try {
        const users = client.db('lab5db').collection('users');
        //console.log(await users.countDocuments());

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
async function update_user(username,password,role,enabled) {
    try {
        const users = client.db('lab5db').collection('users');
        const query = { username: username };
        const update = {
            $set: {
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
        console.log("!null");
        if(user==null)return false;
        else return user;
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
// (async () => {
//     const hktDate = new Date().toLocaleString('en-US', { timeZone: 'Asia/Hong_Kong' });
//     console.log(`HKT: ${hktDate}`);
//     console.log('Server started at http://127.0.0.1:8010');
//     await init_db().catch(console.dir);
//     username_exist('21101256d').then((res) => console.log(res));
// })();
export { validate_user, update_user, fetch_user, username_exist};