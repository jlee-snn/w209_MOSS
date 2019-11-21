/**********************************************************************
 * Main app for w209 MOSS Project
 **********************************************************************/

// update();

// function update () {
  var margin = {top: 20, right: 30, bottom: 40, left: 30},
  width = 600 - margin.left - margin.right,
  height = 500 - margin.top - margin.bottom;

  var x = d3.scaleLinear().range([0, width]);
  var y = d3.scaleBand().range([0, height]).round(0.2);

  var xAxis = d3.axisBottom(x);
  var yAxis = d3.axisLeft(y)
    .tickSize(0)
    .tickPadding(6);


  var data = d3.json("getData", function(error, data) {

    console.log("HERE");

    console.log(data[0]);

    var svg = d3.select("barviz").append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
    .append("g")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    x.domain(d3.extent(data, function(d) { return d.value; })).nice();
    y.domain([data].map(function(d) { return d.word; }));

    svg.selectAll(".bar")
        .data([data])
      .enter().append("rect")
        .attr("class", function(d) { return "bar bar--" + (d.value < 0 ? "negative" : "positive"); })
        .attr("x", function(d) { return x(Math.min(0, d.value)); })
        .attr("y", function(d) { return y(d.word); })
        .attr("width", function(d) { return Math.abs(x(d.value) - x(0)); })
        .attr("height", y.bandwidth())
        ;

    svg.append("g")
        .attr("class", "x axis")
        .attr("transform", "translate(0," + height + ")")
        .call(xAxis);

    svg.append("g")
        .attr("class", "y axis")
        .attr("transform", "translate(" + x(0) + ",0)")
        .call(yAxis);
  });
// } 