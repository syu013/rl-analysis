git push heroku develop:master
heroku ps:scale web=1
heroku open