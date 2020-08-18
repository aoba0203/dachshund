import React from 'react';
import ReactDOM from 'react-dom';
import './index.css';
import App from './App';
import Dashboard from './dashboard/Dashboard'
import * as serviceWorker from './serviceWorker';
// import {startWsServer} from './communication/WsClient';

// import { Provider } from 'react-redux';

// startWsServer()
ReactDOM.render(
  <React.StrictMode>
    {/* <App /> */}
    <Dashboard />
  </React.StrictMode>, 
  document.getElementById('root')
);

// If you want your app to work offline and load faster, you can change
// unregister() to register() below. Note this comes with some pitfalls.
// Learn more about service workers: https://bit.ly/CRA-PWA
serviceWorker.unregister();
