import React from 'react';
import ReactDOM from 'react-dom';
import './index.css';
import App from './App';
import Dashboard from './dashboard/Dashboard';
import ProjectHome from './home/ProjectHome';
import Home from './home/Home'
import Counter from './home/components/Counter'
import * as serviceWorker from './serviceWorker';
// import {startWsServer} from './communication/WsClient';

import CounterContainer from './home/containers/CounterContainer'
import FileUploaderContainer from './home/containers/FileUploaderContainer'
import { Provider } from 'react-redux';
import { createStore } from 'redux';
import rootReducer from './home/modules';
import { composeWithDevTools } from 'redux-devtools-extension';

const store = createStore(rootReducer, composeWithDevTools());

// startWsServer()
ReactDOM.render(
  <React.StrictMode>
    <Provider store={store}>
      {/* <App /> */}
      {/* <Dashboard /> */}
      {/* <ProjectHome /> */}
      {/* <Home />   */}
      {/* <CounterContainer /> */}
      <FileUploaderContainer />
    </Provider>
  </React.StrictMode>, 
  document.getElementById('root')
);

// If you want your app to work offline and load faster, you can change
// unregister() to register() below. Note this comes with some pitfalls.
// Learn more about service workers: https://bit.ly/CRA-PWA
serviceWorker.unregister();

