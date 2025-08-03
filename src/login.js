import express from 'express';
import multer from 'multer';
import fs from 'fs/promises';
import { validate_user, update_user, fetch_user, username_exist} from './userdb.js';
var app = express();
let route = express.Router();
const form = multer();
const user_database = new Map();

// async function init_userdb() {
//   if (user_database.size > 0) {
//     console.log('User database is not empty,');
//     return;
//   }
//   try {
//     const data = await fs.readFile('./users.json', 'utf-8');
//     const userData = JSON.parse(data);
//     userData.forEach((user) => {
//       user_database.set(user.username, user);
//     });
//     console.log(user_database.size);
//   } catch (err) {
//     console.error('error', err);
//   }
// }

// async function validate_user(username, password) {
//   if (user_database.get(username) == null) return false;
//   if (user_database.get(username)['password'] == password) {
//     console.log(user_database.get(username));
//     return user_database.get(username);
//   } else return false;
// }

// async function update_user(username, password, role){
//   const data = await fs.readFile('./users.json', 'utf-8');
//   user_database.set(username, ({"username":username,"password": password, "role": role, "enabled": true}));
//   var userjson=[];
//   try {
//     user_database.forEach((user) => {
//       userjson.push(user);
//     });
//     await fs.writeFile('./users.json', JSON.stringify(userjson), 'utf-8');
//     return true
//   } catch (err) {
//     console.error('error', err);
//     return false;
//   }
// }
route.post('/login', form.none(), async (req, res) => {
  if (req.session.logged) {
    req.session.logged = false; // Reset login status to false
  }
  const { username, password } = req.body; // Extract username and password

  // Validate user credentials
  const user = await validate_user(username, password);
  if (user) {
    console.log(user.username);
    if (!user.enabled) {
      return res.status(401).json({
        status: 'failed',
        message: `User \`${username}\` is currently disabled`,
      });
    }

    // If user is valid and enabled, set session variables
    req.session.username = user.username;
    req.session.role = user.role; // Assuming user role is available
    req.session.logged = true;
    req.session.timestamp = Date.now(); // Current timestamp

    return res.json({
      status: 'success',
      user: {
        username: user.username,
        role: user.role,
      },
    });
  }

  // If validation fails, respond with an appropriate message
  res.status(401).json({
    status: 'failed',
    message: 'Incorrect username and password',
  });
});

// Define the request handler for POST /logout
route.post('/logout', (req, res) => {
  if (req.session.logged) {
    req.session.destroy((err) => {
      if (err) {
        return res.status(500).json({ status: 'failed', message: 'Unable to log out' });
      }
      res.end(); // Send an empty response
    });
  } else {
    res.status(401).json({
      status: 'failed',
      message: 'Unauthorized',
    });
  }
});

// Define the asynchronous request handler for GET /me
route.get('/me', async (req, res) => {
  if (req.session.logged) {
    const user = await fetch_user(req.session.username); // Retrieve user from the in-memory database
    return res.json({
      status: 'success',
      user: {
        username: user.username,
        role: user.role,
      },
    });
  }

  res.status(401).json({
    status: 'failed',
    message: 'Unauthorized',
  });
});
// Define the request handler for POST /register
route.post('/register', form.none(), async(req, res) => {
  const { username, password, role, enabled} = req.body; // Extract username and password
  if (username!=''&&password!=''&&role!='PleaseSelect') {
    var flag=true;
    var criteria;
    if(username.length<3 & flag){
      criteria="Username less than 3 characters";
      flag=false;
    }
    if(password.length<8 & flag){
      criteria="Password less than 8 characters";
      flag=false;
    }
    if(role=="admin"& flag){
      criteria="Role can only be either `student` or `user`";
      flag=false;
    }
    // console.log("!212");
    // console.log(fetch_user(username)=='null');
    if(await fetch_user(username)){
      criteria=`Username ${username} already exists`;
      console.log(criteria);
      flag=false;
    }
    if(!flag){
      return res.status(400).json({
        status: 'failed',
        message: criteria,
      });
    }
    if(await update_user(username, password, role)){
      return res.status(200).json({
        status: 'success',
        user: {
          username: username,
          role: role,
        },
      });
    }
    else{
      return res.status(500).json({
        status: 'failed',
        message: 'Account created but unable to save into the database',
      });
    }
  } else {
    return res.status(400).json({
      status: 'failed',
      message: 'Missing fields',
    });
  }
});

route.post('/update', form.none(), async(req, res) => {
  const { username, password, role, enabled} = req.body; // Extract username and password
  if (username!=''&&password!=''&&role!='PleaseSelect') {
    var flag=true;
    var criteria;
    if(username.length<3 & flag){
      criteria="Username less than 3 characters";
      flag=false;
    }
    if(password.length<8 & flag){
      criteria="Password less than 8 characters";
      flag=false;
    }
    if(role=="admin"& flag){
      criteria="Role can only be either `student` or `user`";
      flag=false;
    }
    // console.log("!212");
    // console.log(fetch_user(username)=='null');
    if(await fetch_user(username)){
      criteria=`Username ${username} already exists`;
      console.log(criteria);
      flag=false;
    }
    if(!flag){
      return res.status(400).json({
        status: 'failed',
        message: criteria,
      });
    }
    if(await update_user(username, password, role)){
      return res.status(200).json({
        status: 'success',
        user: {
          username: username,
          role: role,
        },
      });
    }
    else{
      return res.status(500).json({
        status: 'failed',
        message: 'Account created but unable to save into the database',
      });
    }
  } else {
    return res.status(400).json({
      status: 'failed',
      message: 'Missing fields',
    });
  }
});
export { route };
