import express from 'express';
import session from 'express-session';
import { route as login } from './login.js';
import { route as details } from './event.js';
import { route as payment_page } from './payment.js';
import { route as admin } from './admin.js';
import mongostore from 'connect-mongo';
import client from './dbclient.js';
import fs from 'fs';
const port = process.env.PORT || 4000;
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
app.use('/event', details);
app.use('/admin', admin);
app.use('/pay', payment_page);
app.use(express.static('public'));
app.use('/', express.static('./public'));

app.get('/', (req, res) => {
  if (req.session.logged) {
    return res.redirect('/index.html');
  }
  res.redirect('/login.html');
});


var path = './static';
app.use('/', express.static(path));
// const server = app.listen(8010, () => {
//   const hktDate = new Date().toLocaleString('en-US', { timeZone: 'Asia/Hong_Kong' });
//   console.log(`HKT: ${hktDate}`);
//   console.log('Server started at http://127.0.0.1:8090');
// });
app.listen(port, () => {
  console.log(`Example app listening on port ${port}`)
  console.log('Server started at http://127.0.0.1:4000');
})
export default app;
