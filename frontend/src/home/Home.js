import React from 'react';
import ReactDOM from 'react-dom';
import {createScore} from 'redux';
import {HashRouter as Router, Route } from 'react-router-dom';
import Navigators from './Navigators';
import ProjectHome from './ProjectHome';
import Dashboard from './Dashboard';

class Home extends React.Component{
  constructor(props){
    super(props);
    this.state = {title: 'Title'};
  }

  render(){
    return (
      <Router>
        <Navigators title={this.state.title}>
          <div>
            <Route exact path='/' component={ProjectHome}/>
            <Route exact path='/detail' component={Dashboard}/>
          </div>
        </Navigators>
      </Router>
    )
  }
}

export default Home;