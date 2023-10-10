using Drumwise.Domain.Entities;
using Microsoft.AspNetCore.Identity;

namespace Drumwise.Infrastructure.Identity;

public class ApplicationUser : IdentityUser
{
    public string? Name { get; set; }
    
    public string? Surname { get; set; }

    public List<UserRating> UserRatings { get; set; } = new List<UserRating>();

    public List<Homework> Homeworks { get; set; } = new List<Homework>();

    public List<Lesson> Lessons { get; set; } = new List<Lesson>(); 
    //TODO Add roles
}