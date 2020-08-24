import React from 'react';
import Card from '@material-ui/core/Card';
import CardContent from '@material-ui/core/CardContent';

class Dashboard extends React.Component{  
  render(){
    console.log('Dashboard')
    return (
    <Card>
      <CardContent>
        DashBoard Page
      </CardContent>
    </Card>
    )}
}

export default Dashboard;