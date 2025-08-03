import express from 'express';
import session from 'express-session';
import { route as login } from './login.js';
import mongostore from 'connect-mongo';
import client from './dbclient.js';
import fs from 'fs';
var app = express();
app.use(
  session({
  secret: '21101256d_eie4432_lab5',
  resave: false,
  saveUninitialized: false,
  cookie: { httpOnly: true },
  store: mongostore.create({
  client,
  dbName: 'lab5db',
  collectionName: 'session',
  }),
  })
 );
app.use('/auth', login);

app.use(express.static('public'));

app.get('/myprofile.html', (req, res) => {
  if (req.session.logged) {
    return res.redirect('/myprofile.html');
  }
  res.redirect('/login.html');
});

app.get('/myorder.html', (req, res) => {
  if (req.session.logged) {
    return res.redirect('/myorder.html');
  }
  res.redirect('/login.html');
});

app.get('/', (req, res) => {
  if (req.session.logged) {
    return res.redirect('/index.html');
  }
  res.redirect('/login.html');
});

var path = './static';
app.use('/', express.static(path));
export {app};
// const server = app.listen(8010, () => {
//   const hktDate = new Date().toLocaleString('en-US', { timeZone: 'Asia/Hong_Kong' });
//   console.log(`HKT: ${hktDate}`);
//   console.log('Server started at http://127.0.0.1:8010');
// });
