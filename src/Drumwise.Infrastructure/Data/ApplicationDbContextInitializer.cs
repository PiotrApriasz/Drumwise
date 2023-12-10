using Drumwise.Application.Entities;
using Microsoft.AspNetCore.Builder;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.DependencyInjection;

namespace Drumwise.Infrastructure.Data;

public static class ApplicationDbInitializerExtensions
{
    public static async Task InitializeAppDatabaseAsync(this WebApplication app)
    {
        using var scope = app.Services.CreateScope();
        var initializer = scope.ServiceProvider.GetRequiredService<ApplicationDbContextInitializer>();
        await initializer.InitializeAsync();
        await initializer.SeedAsync();
    }
}

public class ApplicationDbContextInitializer(ApplicationDbContext context)
{
    public async Task InitializeAsync()
    {
        try
        {
            await context.Database.MigrateAsync();
        }
        catch (Exception e)
        {
            Console.WriteLine(e);
            throw;
        }
    }
    
    public async Task SeedAsync()
    {
        // Default data
        // Seed, if necessary
        var guidOne = new Guid().ToString();
        var guidTwo = new Guid().ToString();
        var guidThree = new Guid().ToString();
        
        if (!context.Lessons.Any())
        {
            context.Lessons.Add(new Lesson
            {
                Created = DateTime.Now,
                CreatedBy = guidOne,
                LastModified = DateTime.Now,
                LastModifiedBy = guidOne,
                LessonSubject = "Quarter Notes",
                Exercise = "Play quarter notes for one hour on snare in tempo between 50 and 70"
            });
        }

        if (!context.Homeworks.Any())
        {
            context.Homeworks.Add(new Homework
            {
                Created = DateTime.Now,
                CreatedBy = guidTwo,
                LastModified = DateTime.Now,
                LastModifiedBy = guidTwo,
                AssignedTo = guidThree,
                HomeworkTitle = "Learn new song",
                Exercise = "Choose your favourite song and learn it on drums",
                Deadline = DateTime.Now.AddDays(5)
            });
        }

        if (!context.UserRatings.Any())
        {
            context.UserRatings.Add(new UserRating
            {
                Created = DateTime.Now,
                CreatedBy = guidThree,
                LastModified = DateTime.Now,
                LastModifiedBy = guidThree,
                Mark = 5,
                Comment = "The best teacher",
                AssignedTo = guidTwo
            });
        }
        
        await context.SaveChangesAsync();
    }
}

